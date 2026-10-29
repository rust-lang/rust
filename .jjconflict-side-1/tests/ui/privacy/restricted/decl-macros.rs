#![feature(decl_macro)]

mod m {
    pub macro mac() {
        struct A {}
        pub(self) struct B {} //~ ERROR visibilities can only be restricted to ancestor modules
        pub(in crate::m) struct C {} //~ ERROR visibilities can only be restricted to ancestor modules
    }
}

mod n {
    crate::m::mac!();
}

fn main() {}
