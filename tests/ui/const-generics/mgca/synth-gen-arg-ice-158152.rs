#![feature(min_generic_const_args)]

trait A<T> {}
trait Trait<const N: usize> {}

impl A<[usize; fn_item]> for () {}
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for implementations

fn fn_item(_: impl Trait<usize>) {}
//~^ ERROR: type provided when a constant was expected

fn main() {}
