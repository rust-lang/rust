#![feature(extern_types)]
#![feature(core_intrinsics)]

use std::intrinsics::{align_of_val, size_of_val};

extern "C" {
    type Opaque;
}

const _SIZE: usize = unsafe { size_of_val(&4 as *const i32 as *const Opaque) };
//~^ ERROR `extern type` does not have known layout
const _ALIGN: usize = unsafe { align_of_val(&4 as *const i32 as *const Opaque) };
//~^ ERROR `extern type` does not have known layout

fn main() {}
