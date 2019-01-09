#![allow(unused)]

extern crate core;
pub use core as reexported_core; //~ ERROR `core` is private, and cannot be re-exported
                                 //~^ WARN this was previously accepted

mod foo1 {
    extern crate core;
}

mod foo2 {
    use foo1::core; //~ ERROR `core` is private, and cannot be re-exported
                    //~^ WARN this was previously accepted
    pub mod bar {
        extern crate core;
    }
}

mod baz {
    pub use foo2::bar::core; //~ ERROR `core` is private, and cannot be re-exported
                             //~^ WARN this was previously accepted
}

fn main() {}
