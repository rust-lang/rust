// Make sure we don't ICE when encountering an fn item during lowering in mGCA.

#![feature(min_generic_const_args)]

trait A<T> {}

impl A<[usize; fn_item]> for () {}
//~^ ERROR the constant `fn_item` is not of type `usize`

fn fn_item() {}

fn main() {}
