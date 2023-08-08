// edition:2015
#![feature(decl_macro)]

macro a() {
    extern crate core as my_core;
    mod v {
        // Early resolution.
        use my_core; //~ ERROR unresolved import `my_core`
    }
    mod u {
        // Late resolution.
        fn f() { my_core::mem::drop(0); }
        //~^ ERROR failed to resolve: use of undeclared crate or module `my_core`
    }
}

a!();

mod v {
    // Early resolution.
    use my_core; //~ ERROR unresolved import `my_core`
}
mod u {
    // Late resolution.
    fn f() { my_core::mem::drop(0); }
    //~^ ERROR failed to resolve: use of undeclared crate or module `my_core`
}

fn main() {}
