// compile-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

pub mod two_tuple {
    pub trait T { fn dummy(&self) { } }
    pub struct P<'a>(&'a (T + 'a), &'a (T + 'a));
    pub fn f<'a>(car: &'a T, cdr: &'a T) -> P<'a> {
        P(car, cdr)
    }
}

pub mod two_fields {
    pub trait T { fn dummy(&self) { } }
    pub struct P<'a> { car: &'a (T + 'a), cdr: &'a (T + 'a) }
    pub fn f<'a>(car: &'a T, cdr: &'a T) -> P<'a> {
        P{ car: car, cdr: cdr }
    }
}

fn main() {}
