#![warn(clippy::size_of_in_element_count)]
#![allow(clippy::ptr_offset_with_cast)]

use std::mem::{size_of, size_of_val};
use std::ptr::{
    copy, copy_nonoverlapping, slice_from_raw_parts, slice_from_raw_parts_mut, swap_nonoverlapping, write_bytes,
};
use std::slice::{from_raw_parts, from_raw_parts_mut};

fn main() {
    const SIZE: usize = 128;
    const HALF_SIZE: usize = SIZE / 2;
    const DOUBLE_SIZE: usize = SIZE * 2;
    let mut x = [2u8; SIZE];
    let mut y = [2u8; SIZE];

    // Count is size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping::<u8>(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>()) };
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0])) };

    unsafe { x.as_ptr().copy_to(y.as_mut_ptr(), size_of::<u8>()) };
    unsafe { x.as_ptr().copy_to_nonoverlapping(y.as_mut_ptr(), size_of::<u8>()) };
    unsafe { y.as_mut_ptr().copy_from(x.as_ptr(), size_of::<u8>()) };
    unsafe { y.as_mut_ptr().copy_from_nonoverlapping(x.as_ptr(), size_of::<u8>()) };

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>()) };
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0])) };

    unsafe { y.as_mut_ptr().write_bytes(0u8, size_of::<u8>() * SIZE) };
    unsafe { write_bytes(y.as_mut_ptr(), 0u8, size_of::<u8>() * SIZE) };

    unsafe { swap_nonoverlapping(y.as_mut_ptr(), x.as_mut_ptr(), size_of::<u8>() * SIZE) };

    slice_from_raw_parts_mut(y.as_mut_ptr(), size_of::<u8>() * SIZE);
    slice_from_raw_parts(y.as_ptr(), size_of::<u8>() * SIZE);

    unsafe { from_raw_parts_mut(y.as_mut_ptr(), size_of::<u8>() * SIZE) };
    unsafe { from_raw_parts(y.as_ptr(), size_of::<u8>() * SIZE) };

    unsafe { y.as_mut_ptr().sub(size_of::<u8>()) };
    y.as_ptr().wrapping_sub(size_of::<u8>());
    unsafe { y.as_ptr().add(size_of::<u8>()) };
    y.as_mut_ptr().wrapping_add(size_of::<u8>());
    unsafe { y.as_ptr().offset(size_of::<u8>() as isize) };
    y.as_mut_ptr().wrapping_offset(size_of::<u8>() as isize);

    // Count expression involving multiplication of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * SIZE) };

    // Count expression involving nested multiplications of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), HALF_SIZE * size_of_val(&x[0]) * 2) };

    // Count expression involving divisions of size_of (Should trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE * size_of::<u8>() / 2) };

    // No size_of calls (Should not trigger the lint)
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), SIZE) };

    // Different types for pointee and size_of (Should not trigger the lint)
    unsafe { y.as_mut_ptr().write_bytes(0u8, size_of::<u16>() / 2 * SIZE) };
}
