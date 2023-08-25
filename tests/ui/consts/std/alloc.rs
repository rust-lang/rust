// stderr-per-bitwidth
// Strip out raw byte dumps to make comparison platform-independent:
// normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
// normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*a(lloc)?[0-9]+(\+[a-z0-9]+)?─*╼ )+ *│.*" -> "HEX_DUMP"
use std::alloc::Layout;

// ok
const LAYOUT_VALID: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x08) };

// not ok, since alignment needs to be non-zero.
const LAYOUT_INVALID_ZERO: Layout = unsafe { Layout::from_size_align_unchecked(0x1000, 0x00) };
//~^ ERROR it is undefined behavior to use this value

// not ok, since alignment needs to be a power of two.
const LAYOUT_INVALID_THREE: Layout = unsafe { Layout::from_size_align_unchecked(9, 3) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
