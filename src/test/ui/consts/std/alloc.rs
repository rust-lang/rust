// stderr-per-bitwidth
// ignore-debug (the debug assertions change the error)
use std::alloc::Layout;

// ok
const LAYOUT_VALID: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x08) };

// not ok, since alignment needs to be non-zero.
const LAYOUT_INVALID_ZERO: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x00) };

// not ok, since alignment needs to be a power of two.
const LAYOUT_INVALID_THREE: Layout = unsafe { Layout::from_size_align_unchecked(9, 3) };

fn main() {}
