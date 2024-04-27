//@ known-bug: #117647

#![feature(lifetime_capture_rules_2024)]
#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

use std::ops::Deref;

fn foo(x: Vec<i32>) -> Box<dyn for<'a> Deref<Target = impl ?Sized>> {
    Box::new(x)
}

fn main() {}
