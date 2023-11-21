#![forbid(internal_features)]
// A lang feature and a lib feature.
#![feature(intrinsics, panic_internals)]
//~^ ERROR: internal
//~| ERROR: internal

extern "rust-intrinsic" {
    fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
}

fn main() {}
