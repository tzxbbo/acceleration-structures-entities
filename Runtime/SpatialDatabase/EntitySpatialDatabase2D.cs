using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;

namespace Otz.AccelerationStructures.Entities
{
    public interface ISpatialQueryCollector<T> where T : unmanaged
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void OnVisitCell(in SpatialDatabaseCell cell, in UnsafeList<SpatialDatabase2dElement<T>> elements, out bool shouldEarlyExit);
    }

    [InternalBufferCapacity(0)]
    public struct SpatialDatabaseCell : IBufferElementData
    {
        public int StartIndex;
        public int ElementsCount;
        public int ElementsCapacity;
        public int ExcessElementsCount;
    }

    [InternalBufferCapacity(0)]
    public struct SpatialDatabase2dElement<T> : IBufferElementData where T : unmanaged
    {
        public T Value;
        public float2 Position;
    }

    public struct SpatialDatabaseCellIndex : IComponentData
    {
        public int CellIndex;
    }

    public struct EntitySpatialDatabase2D<T> where T : unmanaged
    {
        public UniformOriginGrid2D Grid;

        public const float ElementsCapacityGrowFactor = 2f;

        public static void Initialize(float halfExtents, int subdivisions, int cellEntriesCapacity,
            ref EntitySpatialDatabase2D<T> spatialDatabase, ref DynamicBuffer<SpatialDatabaseCell> cellsBuffer,
            ref DynamicBuffer<SpatialDatabase2dElement<T>> storageBuffer)
        {
            // Clear
            cellsBuffer.Clear();
            storageBuffer.Clear();
            cellsBuffer.Capacity = 16;
            storageBuffer.Capacity = 16;

            // Init grid
            spatialDatabase.Grid = new UniformOriginGrid2D(halfExtents, subdivisions);

            // Reallocate
            cellsBuffer.Resize(spatialDatabase.Grid.CellCount, NativeArrayOptions.ClearMemory);
            storageBuffer.Resize(spatialDatabase.Grid.CellCount * cellEntriesCapacity, NativeArrayOptions.ClearMemory);

            // Init cells data
            for (int i = 0; i < cellsBuffer.Length; i++)
            {
                SpatialDatabaseCell cell = cellsBuffer[i];
                cell.StartIndex = i * cellEntriesCapacity;
                cell.ElementsCount = 0;
                cell.ElementsCapacity = cellEntriesCapacity;
                cell.ExcessElementsCount = 0;
                cellsBuffer[i] = cell;
            }
        }

        public static void ClearAndResize(ref DynamicBuffer<SpatialDatabaseCell> cellsBuffer,
            ref DynamicBuffer<SpatialDatabase2dElement<T>> storageBuffer)
        {
            int totalDesiredStorage = 0;
            for (int i = 0; i < cellsBuffer.Length; i++)
            {
                SpatialDatabaseCell cell = cellsBuffer[i];
                cell.StartIndex = totalDesiredStorage;

                // Handle calculating an increased max storage for this cell
                cell.ElementsCapacity = math.select(cell.ElementsCapacity,
                    (int)math.ceil((cell.ElementsCapacity + cell.ExcessElementsCount) * ElementsCapacityGrowFactor),
                    cell.ExcessElementsCount > 0);
                totalDesiredStorage += cell.ElementsCapacity;

                // Reset storage
                cell.ElementsCount = 0;
                cell.ExcessElementsCount = 0;

                cellsBuffer[i] = cell;
            }

            storageBuffer.Resize(totalDesiredStorage, NativeArrayOptions.ClearMemory);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddToDatabase(in EntitySpatialDatabase2D<T> spatialDatabase,
            ref UnsafeList<SpatialDatabaseCell> cellsBuffer, ref UnsafeList<SpatialDatabase2dElement<T>> storageBuffer,
            in SpatialDatabase2dElement<T> element)
        {
            int cellIndex = UniformOriginGrid2D.GetCellIndex(in spatialDatabase.Grid, element.Position);
            if (cellIndex >= 0)
            {
                SpatialDatabaseCell cell = cellsBuffer[cellIndex];

                // Check capacity
                if (cell.ElementsCount + 1 > cell.ElementsCapacity)
                {
                    // Remember excess count for resizing next time we clear
                    cell.ExcessElementsCount++;
                }
                else
                {
                    // Add entry at cell index
                    storageBuffer[cell.StartIndex + cell.ElementsCount] = element;
                    cell.ElementsCount++;
                }

                cellsBuffer[cellIndex] = cell;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddToDataBase(in EntitySpatialDatabase2D<T> spatialDatabase,
            ref UnsafeList<SpatialDatabaseCell> cellsBuffer, ref UnsafeList<SpatialDatabase2dElement<T>> storageBuffer,
            in SpatialDatabase2dElement<T> element, int cellIndex)
        {
            if (cellIndex >= 0)
            {
                SpatialDatabaseCell cell = cellsBuffer[cellIndex];

                // Check capacity
                if (cell.ElementsCount + 1 > cell.ElementsCapacity)
                {
                    // Remember excess count for resizing next time we clear
                    cell.ExcessElementsCount++;
                }
                else
                {
                    // Add entry at cell index
                    storageBuffer[cell.StartIndex + cell.ElementsCount] = element;
                    cell.ElementsCount++;
                }

                cellsBuffer[cellIndex] = cell;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static void QueryAABB<K>(in EntitySpatialDatabase2D<T> spatialDatabase,
            in DynamicBuffer<SpatialDatabaseCell> cellsBuffer, in DynamicBuffer<SpatialDatabase2dElement<T>> elementsBuffer,
            float2 center, float2 halfExtents, ref K collector)
            where K : unmanaged, ISpatialQueryCollector<T>
        {
            UnsafeList<SpatialDatabaseCell> cells =
                new UnsafeList<SpatialDatabaseCell>((SpatialDatabaseCell*)cellsBuffer.GetUnsafeReadOnlyPtr(),
                    cellsBuffer.Length);
            UnsafeList<SpatialDatabase2dElement<T>> elements =
                new UnsafeList<SpatialDatabase2dElement<T>>((SpatialDatabase2dElement<T>*)elementsBuffer.GetUnsafeReadOnlyPtr(),
                    elementsBuffer.Length);
            QueryAABB(in spatialDatabase, in cells, in elements, center, halfExtents, ref collector);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void QueryAABB<K>(in EntitySpatialDatabase2D<T> spatialDatabase,
            in UnsafeList<SpatialDatabaseCell> cellsBuffer, in UnsafeList<SpatialDatabase2dElement<T>> elementsBuffer,
            float2 center, float2 halfExtents, ref K collector)
    where K : unmanaged, ISpatialQueryCollector<T>
        {
            float2 aabbMin = center - halfExtents;
            float2 aabbMax = center + halfExtents;
            UniformOriginGrid2D grid = spatialDatabase.Grid;
            if (UniformOriginGrid2D.GetAABBMinMaxCoords(in grid, aabbMin, aabbMax, out int2 minCoords,
                    out int2 maxCoords))
            {
                for (int y = minCoords.y; y <= maxCoords.y; y++)
                {
                    for (int x = minCoords.x; x <= maxCoords.x; x++)
                    {
                        int2 coords = new int2(x, y);
                        int cellIndex = UniformOriginGrid2D.GetCellIndexFromCoords(in grid, coords);
                        SpatialDatabaseCell cell = cellsBuffer[cellIndex];
                        collector.OnVisitCell(in cell, in elementsBuffer,
                            out bool shouldEarlyExit);
                        if (shouldEarlyExit)
                        {
                            return;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static void QueryAABBCellProximityOrder<K>(in EntitySpatialDatabase2D<T> spatialDatabase,
            in DynamicBuffer<SpatialDatabaseCell> cellsBuffer, in DynamicBuffer<SpatialDatabase2dElement<T>> elementsBuffer,
            float2 center, float2 halfExtents, ref K collector)
            where K : unmanaged, ISpatialQueryCollector<T>
        {
            UnsafeList<SpatialDatabaseCell> cells =
                new UnsafeList<SpatialDatabaseCell>((SpatialDatabaseCell*)cellsBuffer.GetUnsafeReadOnlyPtr(),
                    cellsBuffer.Length);
            UnsafeList<SpatialDatabase2dElement<T>> elements =
                new UnsafeList<SpatialDatabase2dElement<T>>((SpatialDatabase2dElement<T>*)elementsBuffer.GetUnsafeReadOnlyPtr(),
                    elementsBuffer.Length);
            QueryAABBCellProximityOrder(in spatialDatabase, in cells, in elements, center, halfExtents, ref collector);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void QueryAABBCellProximityOrder<K>(in EntitySpatialDatabase2D<T> spatialDatabase,
            in UnsafeList<SpatialDatabaseCell> cellsBuffer, in UnsafeList<SpatialDatabase2dElement<T>> elementsBuffer,
            float2 center, float2 halfExtents, ref K collector)
            where K : unmanaged, ISpatialQueryCollector<T>
        {
            float2 aabbMin = center - halfExtents;
            float2 aabbMax = center + halfExtents;
            UniformOriginGrid2D grid = spatialDatabase.Grid;
            if (UniformOriginGrid2D.GetAABBMinMaxCoords(in grid, aabbMin, aabbMax, out int2 minCoords, out int2 maxCoords))
            {
                int2 sourceCoord = UniformOriginGrid2D.GetCellCoordsFromPosition(in grid, center);
                int2 highestCoordDistances = math.max(maxCoords - sourceCoord, sourceCoord - minCoords);
                int maxLayer = math.max(highestCoordDistances.x, highestCoordDistances.y);

                // Iterate layers of cells around the original cell
                for (int l = 0; l <= maxLayer; l++)
                {
                    int2 yRange = new int2(sourceCoord.y - l, sourceCoord.y + l);
                    int2 xRange = new int2(sourceCoord.x - l, sourceCoord.x + l);

                    for (int y = yRange.x; y <= yRange.y; y++)
                    {
                        int yDistToEdge = math.min(y - minCoords.y, maxCoords.y - y); // positive is inside

                        // Skip coords outside of query coords range
                        if (yDistToEdge < 0)
                        {
                            continue;
                        }

                        for (int x = xRange.x; x <= xRange.y; x++)
                        {
                            int xDistToEdge = math.min(x - minCoords.x, maxCoords.x - x); // positive is inside

                            // Skip coords outside of query coords range
                            if (xDistToEdge < 0)
                            {
                                continue;
                            }

                            int2 coords = new int2(x, y);
                            int2 coordDistToCenter = math.abs(coords - sourceCoord);
                            int maxCoordsDist = math.max(coordDistToCenter.x, coordDistToCenter.y);

                            // Skip all inner coords not belonging to the external layer
                            if (maxCoordsDist != l)
                            {
                                x = xRange.y - 1;
                                continue;
                            }

                            int cellIndex = UniformOriginGrid2D.GetCellIndexFromCoords(in grid, coords);
                            SpatialDatabaseCell cell = cellsBuffer[cellIndex];
                            collector.OnVisitCell(in cell, in elementsBuffer,
                                out bool shouldEarlyExit);
                            if (shouldEarlyExit)
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}