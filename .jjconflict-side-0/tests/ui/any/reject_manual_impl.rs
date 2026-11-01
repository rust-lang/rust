#![feature(try_as_dyn)]

use std::any::TryAsDynCompatible;

struct Foo(dyn Iterator<Item = u32>);

impl TryAsDynCompatible<'static> for Foo {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

struct Bar(dyn Iterator<Item = u32>);

impl<'a> TryAsDynCompatible<'a> for Bar {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

struct Baz;

impl<'a> TryAsDynCompatible<'a> for Baz {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

trait Trait {}

impl<'a> TryAsDynCompatible<'a> for dyn Trait {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

impl TryAsDynCompatible<'static> for dyn Iterator<Item = u32> {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted
//~| ERROR: only traits defined in the current crate can be implemented for arbitrary types

fn main() {}
