#![feature(trait_alias)]

mod inner {
    pub trait A { fn foo(&self); }
    pub trait B { fn foo(&self); }

    impl A for u8 {
        fn foo(&self) {}
    }
    impl B for u8 {
        fn foo(&self) {}
    }

    pub trait C = A + B;
}

use inner::C;

fn main() {
    let t = 1u8;
    t.foo(); //~ ERROR E0034

    inner::A::foo(&t); // ok
}
