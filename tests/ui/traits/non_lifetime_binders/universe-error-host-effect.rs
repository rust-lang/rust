//@ compile-flags: -Znext-solver

#![feature(const_trait_impl, non_lifetime_binders, sized_hierarchy)]
#![allow(incomplete_features)]

use std::marker::PointeeSized;

const trait Other<U: PointeeSized, V: PointeeSized>: PointeeSized {}

trait Guard {}

const impl<X: PointeeSized> Other<X, i32> for X {}

impl<X: PointeeSized> Other<u8, u32> for X where u8: Guard {}
//~^ ERROR the trait bound `u8: Guard` is not satisfied

fn foo<U: PointeeSized, V: PointeeSized>()
where
    for<T> T: const Other<U, V>,
{
}

fn bar() {
    foo::<_, _>();
    //~^ ERROR the trait bound `u8: Guard` is not satisfied
}

fn main() {}
