//@ check-fail

#![allow(unexpected_cfgs)] // invalid cfgs

#[cfg(any(foo, foo::bar))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
fn foo1() {}

#[cfg(any(foo::bar, foo))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
fn foo2() {}

#[cfg(all(foo, foo::bar))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
fn foo3() {}

#[cfg(all(foo::bar, foo))]
//~^ ERROR malformed `cfg` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
fn foo4() {}

fn main() {}
