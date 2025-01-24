//@ run-pass
#![allow(non_camel_case_types)]

enum t { foo(Box<isize>), }

pub fn main() {
    let tt = t::foo(Box::new(10));
    match tt { t::foo(_z) => { } }
}
