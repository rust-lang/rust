//@aux-build:proc_macros.rs
#![allow(dead_code)]
#![warn(clippy::redundant_pub_crate)]

mod m1 {
    fn f() {}
    pub(crate) fn g() {} // private due to m1
    //
    //~^^ redundant_pub_crate
    pub fn h() {}

    mod m1_1 {
        fn f() {}
        pub(crate) fn g() {} // private due to m1_1 and m1
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub(crate) mod m1_2 {
        //~^ redundant_pub_crate
        //:^ private due to m1
        fn f() {}
        pub(crate) fn g() {} // private due to m1_2 and m1
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub mod m1_3 {
        fn f() {}
        pub(crate) fn g() {} // private due to m1
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }
}

pub(crate) mod m2 {
    fn f() {}
    pub(crate) fn g() {} // already crate visible due to m2
    //
    //~^^ redundant_pub_crate
    pub fn h() {}

    mod m2_1 {
        fn f() {}
        pub(crate) fn g() {} // private due to m2_1
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub(crate) mod m2_2 {
        //~^ redundant_pub_crate
        //:^ already crate visible due to m2
        fn f() {}
        pub(crate) fn g() {} // already crate visible due to m2_2 and m2
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub mod m2_3 {
        fn f() {}
        pub(crate) fn g() {} // already crate visible due to m2
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }
}

pub mod m3 {
    fn f() {}
    pub(crate) fn g() {} // ok: m3 is exported
    pub fn h() {}

    mod m3_1 {
        fn f() {}
        pub(crate) fn g() {} // private due to m3_1
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub(crate) mod m3_2 {
        //:^ ok
        fn f() {}
        pub(crate) fn g() {} // already crate visible due to m3_2
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub mod m3_3 {
        fn f() {}
        pub(crate) fn g() {} // ok: m3 and m3_3 are exported
        pub fn h() {}
    }
}

mod m4 {
    fn f() {}
    pub(crate) fn g() {} // private: not re-exported by `pub use m4::*`
    //
    //~^^ redundant_pub_crate
    pub fn h() {}

    mod m4_1 {
        fn f() {}
        pub(crate) fn g() {} // private due to m4_1
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub(crate) mod m4_2 {
        //~^ redundant_pub_crate
        //:^ private: not re-exported by `pub use m4::*`
        fn f() {}
        pub(crate) fn g() {} // private due to m4_2
        //
        //~^^ redundant_pub_crate
        pub fn h() {}
    }

    pub mod m4_3 {
        fn f() {}
        pub(crate) fn g() {} // ok: m4_3 is re-exported by `pub use m4::*`
        pub fn h() {}
    }
}

mod m5 {
    pub mod m5_1 {}
    // Test that the primary span isn't butchered for item kinds that don't have an ident.
    pub(crate) use m5_1::*; //~ redundant_pub_crate
    #[rustfmt::skip]
    pub(crate) use m5_1::{*}; //~ redundant_pub_crate
}

pub use m4::*;

mod issue_8732 {
    #[allow(unused_macros)]
    macro_rules! some_macro {
        () => {};
    }

    #[allow(unused_imports)]
    pub(crate) use some_macro; // ok: macro exports are exempt
}

proc_macros::external! {
    mod priv_mod {
        pub(crate) fn dummy() {}
    }
}

fn main() {}
