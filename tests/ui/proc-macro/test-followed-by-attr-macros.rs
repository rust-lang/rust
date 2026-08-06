//@ check-pass
//@ compile-flags: --test
//@ proc-macro: test-macros.rs
//@ edition: 2024

#![feature(macro_attr)]

extern crate test_macros;

fn main() {}

mod proc_macro_attr {
    #[test]
    #[test_macros::identity_attr]
    fn test() {}
}

mod macro_rules_attr {
    macro_rules! repro {
        attr() { $($tt:tt)* } => { $($tt)* }
    }

    #[test]
    #[repro]
    fn test() {}
}
