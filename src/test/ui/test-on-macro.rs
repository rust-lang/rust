// build-pass (FIXME(62277): could be check-pass?)
// compile-flags:--test

#![deny(warnings)]

macro_rules! foo {
    () => (fn foo(){})
}

#[test]
foo!();

fn main(){}
