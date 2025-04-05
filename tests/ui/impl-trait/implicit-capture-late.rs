//@ edition: 2024

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![rustc_variance_of_opaques]

use std::ops::Deref;

fn foo(x: Vec<i32>) -> Box<dyn for<'a> Deref<Target = impl ?Sized>> { //~ ERROR ['a: o]
    //~^ ERROR cannot capture higher-ranked lifetime
    Box::new(x)
}

fn main() {}
