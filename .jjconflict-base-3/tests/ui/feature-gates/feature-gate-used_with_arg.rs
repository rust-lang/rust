#[used(linker)] //~ ERROR `#[used(linker)]` is currently unstable
static mut USED_LINKER: [usize; 1] = [0];

#[used(compiler)] //~ ERROR `#[used(compiler)]` is currently unstable
static mut USED_COMPILER: [usize; 1] = [0];

fn main() {}
