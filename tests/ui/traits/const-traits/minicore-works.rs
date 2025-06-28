//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver -Cpanic=abort
//@ check-pass

#![feature(no_core)]
#![no_std]
#![no_core]
#![feature(const_trait_impl)]

extern crate minicore;
use minicore::*;

struct Custom;
impl const Add for Custom {
    type Output = ();
    fn add(self, _other: Self) {}
}

const fn test_op() {
    let _x = Add::add(1, 2);
    let _y = Custom + Custom;
}

const fn call_indirect<T: [const] Fn()>(t: &T) {
    t()
}

const fn call() {
    call_indirect(&call);
}
