#![feature(extern_types)]
#![feature(core_intrinsics)]
#![feature(const_size_of_val, const_align_of_val)]

use std::intrinsics::{min_align_of_val, size_of_val};

extern "C" {
    type Opaque;
}

const _SIZE: usize = unsafe { size_of_val(&4 as *const i32 as *const Opaque) }; //~ ERROR
const _ALIGN: usize = unsafe { min_align_of_val(&4 as *const i32 as *const Opaque) }; //~ ERROR

fn main() {}
