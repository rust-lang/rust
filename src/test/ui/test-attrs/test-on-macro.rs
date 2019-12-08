// check-pass
// compile-flags:--test

#![deny(warnings)]

macro_rules! foo {
    () => (fn foo(){})
}

#[test]
foo!(); //~ WARNING `#[test]` attribute should not be used on macros

fn main(){}
