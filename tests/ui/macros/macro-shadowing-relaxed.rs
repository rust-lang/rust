//@ build-pass (FIXME(62277): could be check-pass?)
//@ aux-build:macro-in-other-crate.rs

#![feature(decl_macro)]

macro_rules! my_include {() => {
    // Outer
    macro m() {}
    #[macro_use(from_prelude)] extern crate macro_in_other_crate;

    fn inner() {
        // Inner
        macro m() {}
        macro_rules! from_prelude { () => {} }

        // OK, both `m` and `from_prelude` are macro-expanded,
        // but no more macro-expanded than their counterpart from outer scope.
        m!();
        from_prelude!();
    }
}}

my_include!();

fn main() {}
