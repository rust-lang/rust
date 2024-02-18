//@ check-pass
#![allow(dead_code)]

//@ aux-build:legacy_interaction.rs

#![feature(decl_macro)]
#[allow(unused)]

extern crate legacy_interaction;
// ^ defines
// ```rust
//  macro_rules! m {
//     () => {
//         fn f() {} // (1)
//         g() // (2)
//     }
// }
// ```rust

mod def_site {
    // Unless this macro opts out of hygiene, it should resolve the same wherever it is invoked.
    pub macro m2() {
        ::legacy_interaction::m!();
        f(); // This should resolve to (1)
        fn g() {} // We want (2) resolve to this, not to (4)
    }
}

mod use_site {
    fn test() {
        fn f() -> bool { true } // (3)
        fn g() -> bool { true } // (4)

        ::def_site::m2!();

        let _: bool = f(); // This should resolve to (3)
        let _: bool = g(); // This should resolve to (4)
    }
}

fn main() {}
