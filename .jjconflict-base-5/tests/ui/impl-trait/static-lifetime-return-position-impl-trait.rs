//@ check-pass

#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params)]

pub struct Element;

pub trait Node {
    fn elements<const T: &'static str>(&self) -> impl Iterator<Item = Element>;
}

fn main() {}
