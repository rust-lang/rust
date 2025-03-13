#![feature(ptr_metadata)]
#![feature(trivial_bounds)]

fn return_str()
where
    str: std::ptr::Pointee<Metadata = str>,
{
    [(); { let _a: Option<&str> = None; 0 }];
    //~^ ERROR evaluation of constant value failed
    //~^^ WARN cannot use constants which depend on trivially-false where clauses
    //~| WARN this was previously accepted by the compiler but is being phased out
}

fn main() {}
