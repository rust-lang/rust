// normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
// normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
#![feature(const_mut_refs, const_refs_to_static)]
#![feature(raw_ref_op)]

// This file checks that our dynamic checks catch things that the static checks miss.
// We do not have static checks for these, because we do not look into function bodies.
// We treat all functions as not returning a mutable reference, because there is no way to
// do that without causing the borrow checker to complain (see the B4/helper test in
// mut_ref_in_final.rs).

static mut BUFFER: i32 = 42;

const fn helper() -> Option<&'static mut i32> { unsafe {
    Some(&mut *std::ptr::addr_of_mut!(BUFFER))
} }
const A: Option<&mut i32> = helper(); //~ ERROR it is undefined behavior to use this value
//~^ encountered mutable reference
static A_STATIC: Option<&mut i32> = helper(); //~ ERROR it is undefined behavior to use this value
//~^ encountered mutable reference

fn main() {}
