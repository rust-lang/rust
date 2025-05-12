//@ edition:2018
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
        //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `my_core`
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
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `my_core`
}

fn main() {}
