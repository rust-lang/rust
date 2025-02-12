//@ compile-flags: --crate-type=lib
//@ revisions: e2024 future
//@[e2024] edition: 2024
//@[e2024] check-fail
//@[future] compile-flags: -Zunstable-options
//@[future] edition: future
//@[future] check-pass
#![feature(sized_hierarchy)]

pub fn metasized<T: MetaSized>() {}
//[e2024]~^ ERROR cannot find trait `MetaSized` in this scope
pub fn pointeesized<T: PointeeSized>() {}
//[e2024]~^ ERROR cannot find trait `PointeeSized` in this scope
