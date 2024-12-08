#![feature(used_with_arg)]

#[used(compiler, linker)] //~ expected `used`, `used(compiler)` or `used(linker)`
static mut USED_COMPILER_LINKER: [usize; 1] = [0];

fn main() {}
