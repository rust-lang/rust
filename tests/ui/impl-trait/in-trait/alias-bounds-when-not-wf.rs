//@ compile-flags: -Znext-solver

#![feature(lazy_type_alias)]
//~^ WARN the feature `lazy_type_alias` is incomplete

trait Foo {}

type A<T: Foo> = T;

struct W<T>(T);

// For `W<A<usize>>` to be WF, `A<usize>: Sized` must hold. However, when assembling
// alias bounds for `A<usize>`, we try to normalize it, but it doesn't hold because
// `usize: Foo` doesn't hold. Therefore we ICE, because we don't expect to still
// encounter weak types in `assemble_alias_bound_candidates_recur`.
fn hello(_: W<A<usize>>) {}
//~^ ERROR the trait bound `usize: Foo` is not satisfied
//~| ERROR the trait bound `usize: Foo` is not satisfied
//~| ERROR the trait bound `usize: Foo` is not satisfied

fn main() {}
