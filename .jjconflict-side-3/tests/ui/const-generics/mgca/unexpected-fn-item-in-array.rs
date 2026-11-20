// Make sure we don't ICE when encountering an fn item during lowering in mGCA.

#![feature(gca_min_const_items, gca_macroless_args)]

trait A<T> {}

impl A<[usize; fn_item]> for () {}
//~^ ERROR function items cannot be used as const args

fn fn_item() {}

fn main() {}
