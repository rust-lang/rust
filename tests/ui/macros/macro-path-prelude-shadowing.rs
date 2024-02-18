//@ aux-build:macro-in-other-crate.rs

#![feature(decl_macro)]

macro_rules! add_macro_expanded_things_to_macro_prelude {() => {
    #[macro_use]
    extern crate macro_in_other_crate;
}}

add_macro_expanded_things_to_macro_prelude!();

mod m1 {
    fn check() {
        inline!(); // OK. Theoretically ambiguous, but we do not consider built-in attributes
                   // as candidates for non-attribute macro invocations to avoid regressions
                   // on stable channel
    }
}

mod m2 {
    pub mod std {
        pub macro panic() {}
    }
}

mod m3 {
    use m2::*; // glob-import user-defined `std`
    fn check() {
        std::panic!(); //~ ERROR `std` is ambiguous
    }
}

fn main() {}
