// check-pass

#![feature(decl_macro)]

macro mac() {
    mod m {
        fn f() {
            std::mem::drop(0); // OK (extern prelude)
            drop(0); // OK (stdlib prelude)
        }
    }
}

mac!();

fn main() {}
