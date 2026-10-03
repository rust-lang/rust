//! Regression test for <https://github.com/rust-lang/rust/issues/28279>.
//! Region variables escaped comparison for common supertype, which led
//! `Rc<Fn(&T)>` and `*mut Fn(&T)` to break.
//@ check-pass

#![allow(dead_code)]
use std::rc::Rc;

fn test1() -> Rc<dyn for<'a> Fn(&'a usize) + 'static> {
    if let Some(_) = Some(1) {
        loop{}
    } else {
        loop{}
    }
}

fn test2() -> *mut (dyn for<'a> Fn(&'a usize) + 'static) {
    if let Some(_) = Some(1) {
        loop{}
    } else {
        loop{}
    }
}

fn main() {}
