#![feature(const_trait_bound_opt_out)]
#![allow(bare_trait_objects)]
#![allow(incomplete_features)]

struct S;
trait T {}
impl T for S {}

// An inherent impl for the trait object `?const T`.
impl ?const T {}
//~^ ERROR `?const` is not permitted in trait objects

fn trait_object() -> &'static dyn ?const T { &S }
//~^ ERROR `?const` is not permitted in trait objects

fn trait_object_in_apit(_: impl IntoIterator<Item = Box<dyn ?const T>>) {}
//~^ ERROR `?const` is not permitted in trait objects

fn main() {}
