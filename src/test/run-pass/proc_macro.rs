// aux-build:proc_macro_def.rs
// ignore-cross-compile

#![feature(proc_macro_hygiene)]

extern crate proc_macro_def;

use proc_macro_def::{attr_tru, attr_identity, identity, ret_tru, tru};

#[attr_tru]
fn f1() -> bool {
    return false;
}

#[attr_identity]
fn f2() -> bool {
    return identity!(true);
}

fn f3() -> identity!(bool) {
    ret_tru!();
}

fn f4(x: bool) -> bool {
    match x {
        identity!(true) => false,
        identity!(false) => true,
    }
}

fn main() {
    assert!(f1());
    assert!(f2());
    assert!(tru!());
    assert!(f3());
    assert!(identity!(5 == 5));
    assert!(f4(false));
}
