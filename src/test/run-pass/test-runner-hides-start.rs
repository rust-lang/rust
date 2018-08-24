// compile-flags: --test

#![feature(start)]

#[start]
fn start(_: isize, _: *const *const u8) -> isize { panic!(); }
