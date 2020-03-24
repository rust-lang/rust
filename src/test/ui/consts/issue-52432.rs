#![feature(const_raw_ptr_to_usize_cast)]

fn main() {
    [(); &(static |x| {}) as *const _ as usize];
    //~^ ERROR: closures cannot be static
    //~| ERROR: type annotations needed
    [(); &(static || {}) as *const _ as usize];
    //~^ ERROR: closures cannot be static
}
