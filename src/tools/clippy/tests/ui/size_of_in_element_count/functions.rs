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
    let mut x = [2u16; SIZE];
    let mut y = [2u16; SIZE];

    // Count is size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping::<u16>(x.as_ptr(), y.as_mut_ptr(), size_of::<u16>()) };
    //~^ size_of_in_element_count

    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0])) };
    //~^ size_of_in_element_count

    unsafe { x.as_ptr().copy_to(y.as_mut_ptr(), size_of::<u16>()) };
    //~^ size_of_in_element_count

    unsafe { x.as_ptr().copy_to_nonoverlapping(y.as_mut_ptr(), size_of::<u16>()) };
    //~^ size_of_in_element_count

    unsafe { y.as_mut_ptr().copy_from(x.as_ptr(), size_of::<u16>()) };
    //~^ size_of_in_element_count

    unsafe { y.as_mut_ptr().copy_from_nonoverlapping(x.as_ptr(), size_of::<u16>()) };
    //~^ size_of_in_element_count

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of::<u16>()) };
    //~^ size_of_in_element_count

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0])) };
    //~^ size_of_in_element_count

    unsafe { swap_nonoverlapping(y.as_mut_ptr(), x.as_mut_ptr(), size_of::<u16>() * SIZE) };
    //~^ size_of_in_element_count

    slice_from_raw_parts_mut(y.as_mut_ptr(), size_of::<u16>() * SIZE);
    //~^ size_of_in_element_count

    slice_from_raw_parts(y.as_ptr(), size_of::<u16>() * SIZE);
    //~^ size_of_in_element_count

    unsafe { from_raw_parts_mut(y.as_mut_ptr(), size_of::<u16>() * SIZE) };
    //~^ size_of_in_element_count

    unsafe { from_raw_parts(y.as_ptr(), size_of::<u16>() * SIZE) };
    //~^ size_of_in_element_count

    unsafe { y.as_mut_ptr().sub(size_of::<u16>()) };
    //~^ size_of_in_element_count

    y.as_ptr().wrapping_sub(size_of::<u16>());
    //~^ size_of_in_element_count

    unsafe { y.as_ptr().add(size_of::<u16>()) };
    //~^ size_of_in_element_count

    y.as_mut_ptr().wrapping_add(size_of::<u16>());
    //~^ size_of_in_element_count

    unsafe { y.as_ptr().offset(size_of::<u16>() as isize) };
    //~^ size_of_in_element_count

    y.as_mut_ptr().wrapping_offset(size_of::<u16>() as isize);
    //~^ size_of_in_element_count
}
