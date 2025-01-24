//@ run-pass
//@ ignore-emscripten

#![feature(extern_types)]
#![feature(repr_simd)]

use std::ptr::NonNull;

extern "C" {
    type Extern;
}

#[repr(simd)]
struct S<T>([T; 4]);

#[inline(never)]
fn identity<T>(v: T) -> T {
    v
}

fn main() {
    let _v: S<Option<NonNull<Extern>>> = identity(S([None; 4]));
}
