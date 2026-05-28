//@ check-pass

#![feature(used_with_arg)]

#[used(linker)]
#[no_mangle] // accidentally detected as `used(compiler)`
pub static GLOB: usize = 0;

fn main() {}
