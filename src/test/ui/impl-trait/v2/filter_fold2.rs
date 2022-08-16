// check-pass

#![feature(return_position_impl_trait_v2)]

fn foo<T: FnMut(&u32)>() -> impl Sized {}

fn main() {}
