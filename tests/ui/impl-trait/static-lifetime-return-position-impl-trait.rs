// check-pass

#![allow(incomplete_features)]
#![feature(adt_const_params, return_position_impl_trait_in_trait)]

pub struct Element;

pub trait Node {
    fn elements<const T: &'static str>(&self) -> impl Iterator<Item = Element>;
}

fn main() {}
