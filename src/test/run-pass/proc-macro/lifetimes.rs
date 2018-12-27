#![allow(unused_variables)]
// aux-build:lifetimes.rs

extern crate lifetimes;
use lifetimes::*;

lifetimes_bang! {
    fn bang<'a>() -> &'a u8 { &0 }
}

#[lifetimes_attr]
fn attr<'a>() -> &'a u8 { &1 }

#[derive(Lifetimes)]
pub struct Lifetimes<'a> {
    pub field: &'a u8,
}

fn main() {
    assert_eq!(bang::<'static>(), &0);
    assert_eq!(attr::<'static>(), &1);
    let l1 = Lifetimes { field: &0 };
    let l2 = m::Lifetimes { field: &1 };
}
