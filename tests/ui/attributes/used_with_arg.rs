#![deny(unused_attributes)]
#![feature(used_with_arg)]

#[used(linker)]
static mut USED_LINKER: [usize; 1] = [0];

#[used(compiler)]
static mut USED_COMPILER: [usize; 1] = [0];

#[used(compiler)]
#[used(linker)]
static mut USED_COMPILER_LINKER2: [usize; 1] = [0];

#[used(compiler)]
#[used(linker)]
#[used(compiler)] //~ ERROR unused attribute
#[used(linker)] //~ ERROR unused attribute
static mut USED_COMPILER_LINKER3: [usize; 1] = [0];

#[used(compiler)]
#[used]
static mut USED_WITHOUT_ATTR1: [usize; 1] = [0];

#[used(linker)]
#[used] //~ ERROR unused attribute
static mut USED_WITHOUT_ATTR2: [usize; 1] = [0];

fn main() {}
