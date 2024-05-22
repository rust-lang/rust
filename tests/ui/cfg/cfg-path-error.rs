//@ check-fail

#![allow(unexpected_cfgs)] // invalid cfgs

#[cfg(any(foo, foo::bar))]
//~^ERROR `cfg` predicate key must be an identifier
fn foo1() {}

#[cfg(any(foo::bar, foo))]
//~^ERROR `cfg` predicate key must be an identifier
fn foo2() {}

#[cfg(all(foo, foo::bar))]
//~^ERROR `cfg` predicate key must be an identifier
fn foo3() {}

#[cfg(all(foo::bar, foo))]
//~^ERROR `cfg` predicate key must be an identifier
fn foo4() {}

fn main() {}
