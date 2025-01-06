#![feature(trivial_bounds)]

fn return_str()
where
    str: Sized,
{
    [(); { let _a: Option<str> = None; 0 }];
    //~^ ERROR evaluation of constant value failed
}

fn main() {}
