#![feature(try_as_dyn)]

use std::any::TryAsDynCompatible;

struct Foo(dyn Iterator<Item = u32>);

impl TryAsDynCompatible<u32> for Foo {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

struct Bar(dyn Iterator<Item = u32>);

impl<T> TryAsDynCompatible<T> for Bar {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

struct Baz;

impl<T> TryAsDynCompatible<T> for Baz {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

trait Trait {}

impl<T> TryAsDynCompatible<T> for dyn Trait {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted

impl TryAsDynCompatible<u32> for dyn Iterator<Item = u32> {}
//~^ ERROR: explicit impls for the `TryAsDynCompatible` trait are not permitted
//~| ERROR: only traits defined in the current crate can be implemented for arbitrary types

fn main() {}
