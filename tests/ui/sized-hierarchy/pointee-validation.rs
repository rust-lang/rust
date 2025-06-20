// Test that despite us dropping `PointeeSized` bounds during HIR ty lowering
// we still validate it first.
// issue: <https://github.com/rust-lang/rust/issues/142718>
#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

struct Test where (): const PointeeSized<(), Undefined = ()>;
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR associated type `Undefined` not found for `PointeeSized`
//~| ERROR `const` can only be applied to `#[const_trait]` traits
//~| ERROR const trait impls are experimental

fn main() {}
