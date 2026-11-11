// Test that despite us dropping `PointeeSized` bounds during HIR ty lowering
// we still validate it first.
// issue: <https://github.com/rust-lang/rust/issues/142718>
#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

struct T where (): PointeeSized<(), Undefined = ()>;
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR associated type `Undefined` not found for `PointeeSized`

const fn test<T, U>() where T: const PointeeSized, U: [const] PointeeSized {}
//~^ ERROR `const` can only be applied to `const` traits
//~| ERROR `const` can only be applied to `const` traits
//~| ERROR const trait impls are experimental
//~| ERROR `[const]` can only be applied to `const` traits
//~| ERROR `[const]` can only be applied to `const` traits
//~| ERROR const trait impls are experimental

fn main() {}
