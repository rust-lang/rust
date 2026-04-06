// Regression test for <https://github.com/rust-lang/rust/issues/154900>.
//
//@ check-pass
#![feature(never_type)]
#![crate_type = "lib"]
#![warn(unreachable_code)]

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
//~^ warn: unreachable
pub struct S(!);
//~^ warn: unreachable

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
//~^ warn: unreachable
pub struct S2 {
    f: !,
    //~^ warn: unreachable
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
//~^ warn: unreachable
//~| warn: unreachable
pub enum E {
    E2(!),
    E3 { f: ! },
}
