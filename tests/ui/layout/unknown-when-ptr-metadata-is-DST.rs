#![feature(ptr_metadata)]
#![feature(trivial_bounds)]

fn return_str()
where
    str: std::ptr::Pointee<Metadata = str>,
{
    [(); { let _a: Option<&str> = None; 0 }];
    //~^ ERROR the type `str` has an unknown layout
}

fn main() {}
