#![feature(ptr_metadata)]
#![feature(trivial_bounds)]

fn return_str()
where
    str: std::ptr::Pointee<Metadata = str>,
{
    [(); { let _a: Option<&str> = None; 0 }];
    //~^ ERROR entering unreachable code
    //~| NOTE evaluation of `return_str::{constant#0}` failed here
}

fn main() {}
