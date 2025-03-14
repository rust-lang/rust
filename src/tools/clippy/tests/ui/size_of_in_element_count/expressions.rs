#![warn(clippy::size_of_in_element_count)]
#![allow(clippy::ptr_offset_with_cast)]

use std::mem::{size_of, size_of_val};
use std::ptr::{copy, copy_nonoverlapping, write_bytes};

fn main() {
    const SIZE: usize = 128;
    const HALF_SIZE: usize = SIZE / 2;
    const DOUBLE_SIZE: usize = SIZE * 2;
    let mut x = [2u16; SIZE];
    let mut y = [2u16; SIZE];

    // Count expression involving multiplication of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u16>() * SIZE) };
    //~^ size_of_in_element_count

    // Count expression involving nested multiplications of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), HALF_SIZE * size_of_val(&x[0]) * 2) };
    //~^ size_of_in_element_count

    // Count expression involving divisions of size_of (Should trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE * size_of::<u16>() / 2) };
    //~^ size_of_in_element_count

    // Count expression involving divisions by size_of (Should not trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE / size_of::<u16>()) };

    // Count expression involving divisions by multiple size_of (Should not trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE / (2 * size_of::<u16>())) };

    // Count expression involving recursive divisions by size_of (Should trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE / (2 / size_of::<u16>())) };
    //~^ size_of_in_element_count

    // No size_of calls (Should not trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), SIZE) };
}
