// Regression test for <https://github.com/rust-lang/rust/issues/154900>.
//
//@ check-pass
#![crate_type = "lib"]
#![warn(unreachable_code)]

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct S(!);

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct S2 {
    f: !,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum E {
    E2(!),
    E3 { f: ! },
}
