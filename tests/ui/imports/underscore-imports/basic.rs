//@ check-pass
//@ aux-build:underscore-imports.rs

#![warn(unused_imports, unused_extern_crates)]

#[macro_use]
extern crate underscore_imports as _;

do_nothing!(); // OK

struct S;

mod m {
    pub trait Tr1 {
        fn tr1_is_in_scope(&self) {}
    }
    pub trait Tr2 {
        fn tr2_is_in_scope(&self) {}
    }

    impl Tr1 for crate::S {}
    impl Tr2 for crate::S {}
}

mod unused {
    use crate::m::Tr1 as _; //~ WARN unused import
    use crate::S as _; //~ WARN unused import
    extern crate core as _; // OK
}

mod outer {
    mod middle {
        pub use crate::m::Tr1 as _;
        pub use crate::m::Tr2 as _; // OK, no name conflict
        struct Tr1; // OK, no name conflict
        fn check() {
            // Both traits are in scope
            crate::S.tr1_is_in_scope();
            crate::S.tr2_is_in_scope();
        }

        mod inner {
            // `_` imports are fetched by glob imports
            use super::*;
            fn check() {
                // Both traits are in scope
                crate::S.tr1_is_in_scope();
                crate::S.tr2_is_in_scope();
            }
        }
    }

    // `_` imports are fetched by glob imports
    use self::middle::*;
    fn check() {
        // Both traits are in scope
        crate::S.tr1_is_in_scope();
        crate::S.tr2_is_in_scope();
    }
}

fn main() {}
