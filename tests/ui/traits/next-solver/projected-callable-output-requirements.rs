//@ check-fail
//@ compile-flags: -Znext-solver=globally

#![feature(checked_type_aliases)]
#![allow(incomplete_features, dead_code)]

// Normalizing an output must not discard its lifetime requirements.
type Restricted<'a> = () where 'a: 'static;

fn checked_alias<F: for<'a> Fn() -> Restricted<'a>>() {}
//~^ ERROR binding for associated type `Output` references lifetime `'a`

trait Family {
    type View<'a> where 'a: 'static;
}

impl Family for () {
    type View<'a> = () where 'a: 'static;
}

fn projection<F: for<'a> Fn() -> <() as Family>::View<'a>>() {}
//~^ ERROR binding for associated type `Output` references lifetime `'a`

fn main() {}
