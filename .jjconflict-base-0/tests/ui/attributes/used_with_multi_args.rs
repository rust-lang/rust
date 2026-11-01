#![feature(used_with_arg)]

#[used(compiler, linker)] //~ ERROR malformed `used` attribute input
static mut USED_COMPILER_LINKER: [usize; 1] = [0];

fn main() {}
