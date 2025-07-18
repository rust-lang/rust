//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "( 0x[0-9a-f][0-9a-f] │)? ([0-9a-f][0-9a-f] |__ |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> " HEX_DUMP"
//@ normalize-stderr: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

use std::sync::Mutex;

// This file checks that our dynamic checks catch things that the static checks miss.
// We do not have static checks for these, because we do not look into function bodies.
// We treat all functions as not returning a mutable reference, because there is no way to
// do that without causing the borrow checker to complain (see the B4/helper test in
// mut_ref_in_final.rs).

static mut BUFFER: i32 = 42;

const fn helper() -> Option<&'static mut i32> { unsafe {
    Some(&mut *std::ptr::addr_of_mut!(BUFFER))
} }
const MUT: Option<&mut i32> = helper(); //~ ERROR encountered mutable reference

const fn helper_int2ptr() -> Option<&'static mut i32> { unsafe {
    // Undefined behaviour (integer as pointer), who doesn't love tests like this.
    Some(&mut *(42 as *mut i32))
} }
const INT2PTR: Option<&mut i32> = helper_int2ptr(); //~ ERROR encountered a dangling reference
static INT2PTR_STATIC: Option<&mut i32> = helper_int2ptr(); //~ ERROR encountered a dangling reference

const fn helper_dangling() -> Option<&'static mut i32> { unsafe {
    // Undefined behaviour (dangling pointer), who doesn't love tests like this.
    Some(&mut *(&mut 42 as *mut i32))
} }
const DANGLING: Option<&mut i32> = helper_dangling(); //~ ERROR dangling reference
static DANGLING_STATIC: Option<&mut i32> = helper_dangling(); //~ ERROR dangling reference

// These are fine! Just statics pointing to mutable statics, nothing fundamentally wrong with this.
static MUT_STATIC: Option<&mut i32> = helper();
static mut MUT_ARRAY: &mut [u8] = &mut [42];
static MUTEX: Mutex<&mut [u8]> = Mutex::new(unsafe { &mut *MUT_ARRAY });

fn main() {}
