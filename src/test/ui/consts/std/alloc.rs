use std::alloc::Layout;

// ok
const LAYOUT_VALID: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x08) };

// not ok, since alignment needs to be non-zero.
const LAYOUT_INVALID: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x00) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
