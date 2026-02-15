#![feature(extern_types)]
#![feature(core_intrinsics)]
#![feature(ptr_alignment_type)]

use std::intrinsics::{align_of_val, size_of_val};
use std::ptr::Alignment;

extern "C" {
    type Opaque;
}

const _SIZE: usize = unsafe { size_of_val(&4 as *const i32 as *const Opaque) };
//~^ ERROR: the size for values of type `Opaque` cannot be known
const _ALIGN: Alignment = unsafe { align_of_val(&4 as *const i32 as *const Opaque) };
//~^ ERROR: the size for values of type `Opaque` cannot be known

fn main() {}
