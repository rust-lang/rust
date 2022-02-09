#![feature(used_with_arg)]

#[used(linker)]
static mut USED_LINKER: [usize; 1] = [0];

#[used(compiler)]
static mut USED_COMPILER: [usize; 1] = [0];

#[used(compiler)] //~ ERROR `used(compiler)` and `used(linker)` can't be used together
#[used(linker)]
static mut USED_COMPILER_LINKER2: [usize; 1] = [0];

#[used(compiler)] //~ ERROR `used(compiler)` and `used(linker)` can't be used together
#[used(linker)]
#[used(compiler)]
#[used(linker)]
static mut USED_COMPILER_LINKER3: [usize; 1] = [0];

fn main() {}
