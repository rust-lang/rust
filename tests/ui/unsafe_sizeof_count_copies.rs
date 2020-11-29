#![warn(clippy::unsafe_sizeof_count_copies)]

use std::mem::{size_of, size_of_val};
use std::ptr::{copy, copy_nonoverlapping};

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

    // Count expression involving multiplication of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * SIZE) };
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0]) * SIZE) };

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * SIZE) };
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0]) * SIZE) };

    // Count expression involving nested multiplications of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * HALF_SIZE * 2) };
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), HALF_SIZE * size_of_val(&x[0]) * 2) };

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * SIZE * HALF_SIZE) };
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0]) * HALF_SIZE * 2) };

    // Count expression involving divisions of size_of (Should trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u8>() * DOUBLE_SIZE / 2) };
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE / 2 * size_of_val(&x[0])) };

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), DOUBLE_SIZE * size_of::<u8>() / 2) };
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of_val(&x[0]) * DOUBLE_SIZE / 2) };

    // No size_of calls (Should not trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), SIZE) };
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), SIZE) };

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), SIZE) };
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), SIZE) };

    // Different types for pointee and size_of (Should not trigger the lint)
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of::<u16>() / 2 * SIZE) };
    unsafe { copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), size_of_val(&0u16) / 2 * SIZE) };

    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of::<u16>() / 2 * SIZE) };
    unsafe { copy(x.as_ptr(), y.as_mut_ptr(), size_of_val(&0u16) / 2 * SIZE) };
}
