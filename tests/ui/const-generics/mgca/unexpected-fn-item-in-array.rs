// Make sure we don't ICE when encountering an fn item during lowering in mGCA.

#![feature(min_generic_const_args, macroless_generic_const_args)]

trait A<T> {}

impl A<[usize; fn_item]> for () {}
//~^ ERROR function items cannot be used as const args

fn fn_item() {}

fn main() {}
