//! Regression test for https://github.com/rust-lang/rust/issues/10902

//@ check-pass
#![allow(dead_code)]

pub mod two_tuple {
    pub trait T { fn dummy(&self) { } }
    pub struct P<'a>(&'a (dyn T + 'a), &'a (dyn T + 'a));
    pub fn f<'a>(car: &'a dyn T, cdr: &'a dyn T) -> P<'a> {
        P(car, cdr)
    }
}

pub mod two_fields {
    pub trait T { fn dummy(&self) { } }
    pub struct P<'a> { car: &'a (dyn T + 'a), cdr: &'a (dyn T + 'a) }
    pub fn f<'a>(car: &'a dyn T, cdr: &'a dyn T) -> P<'a> {
        P{ car: car, cdr: cdr }
    }
}

fn main() {}
