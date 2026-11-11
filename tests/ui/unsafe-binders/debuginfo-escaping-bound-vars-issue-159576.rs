//@ check-pass
//@ compile-flags: --crate-type=lib -Cdebuginfo=2

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

pub fn woof() -> unsafe<'a, 'b> &'b Box<dyn Fn(Box<dyn Fn() -> &'a isize>)> {
    todo!()
}
