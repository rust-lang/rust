// stderr-per-bitwidth
// ignore-debug (the debug assertions change the error)
use std::alloc::Layout;

// ok
const LAYOUT_VALID: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x08) };

// not ok, since alignment needs to be non-zero.
const LAYOUT_INVALID_ZERO: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x00) };
//~^ ERROR it is undefined behavior to use this value

// not ok, since alignment needs to be a power of two.
const LAYOUT_INVALID_THREE: Layout = unsafe { Layout::from_size_align_unchecked(9, 3) };
//~^ ERROR it is undefined behavior to use this value

// not ok, since size needs to be no more than `isize::MAX`
const LAYOUT_SIZE_NEGATIVE_ONE: Layout = unsafe { Layout::from_size_align_unchecked(-1 as _, 1) };
//~^ ERROR it is undefined behavior to use this value

// not ok, since size needs to be no more than `isize::MAX`
const LAYOUT_SIZE_HIGH_BIT: Layout = unsafe { Layout::from_size_align_unchecked(SIZE_MAX + 1, 1) };
//~^ ERROR it is undefined behavior to use this value

const SIZE_MAX: usize = isize::MAX as usize;

fn main() {}
