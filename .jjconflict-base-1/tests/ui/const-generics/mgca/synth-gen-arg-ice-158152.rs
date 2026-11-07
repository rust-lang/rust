#![feature(gca_min_const_items, gca_macroless_args)]

trait A<T> {}
trait Trait<const N: usize> {}

impl A<[usize; fn_item]> for () {}
//~^ ERROR function items cannot be used as const args

fn fn_item(_: impl Trait<usize>) {}
//~^ ERROR: type provided when a constant was expected

fn main() {}
