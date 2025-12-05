//@ run-pass

use std::cell::Cell;

fn main() {
    let slice: &mut [i32] = &mut [1, 2, 3];
    let cell_slice: &Cell<[i32]> = Cell::from_mut(slice);
    let slice_cell: &[Cell<i32>] = cell_slice.as_slice_of_cells();

    assert_eq!(slice_cell.len(), 3);

    let mut array: [i32; 3] = [1, 2, 3];
    let cell_array: &Cell<[i32; 3]> = Cell::from_mut(&mut array);
    let array_cell: &[Cell<i32>; 3] = cell_array.as_array_of_cells();

    array_cell[0].set(99);
    assert_eq!(array, [99, 2, 3]);
}
