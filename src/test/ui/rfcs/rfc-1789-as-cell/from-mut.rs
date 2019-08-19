// run-pass

use std::cell::Cell;

fn main() {
    let slice: &mut [i32] = &mut [1, 2, 3];
    let cell_slice: &Cell<[i32]> = Cell::from_mut(slice);
    let slice_cell: &[Cell<i32>] = cell_slice.as_slice_of_cells();

    assert_eq!(slice_cell.len(), 3);
}
