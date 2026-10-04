#![deny(unused)]

mod foo {
    fn f() {}

    mod m1 {
        pub(super) use super::f; //~ ERROR unused
    }

    mod m2 {
        #[allow(unused)]
        use super::m1::*; // (despite this glob import)
    }

    mod m3 {
        pub(super) use super::f; // Check that this is counted as used (cf. issue #36249).
    }

    pub mod m4 {
        use super::m3::*;
        pub fn g() { f(); }
    }
}

fn main() {
    foo::m4::g();
}
