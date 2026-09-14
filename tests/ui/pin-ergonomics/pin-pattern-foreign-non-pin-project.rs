//@ edition:2024
//@ aux-build:non_pin_project.rs
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// Regression test for #157634, exercising the diagnostic good-practice raised in the #157542
// review: the projection error must be emitted even when the projected-through type comes from
// another crate and therefore has no local span. `ProjectOnNonPinProjectType` carries its
// `def_span`/`sugg_span` as `Option<Span>`, so for a foreign type the "type defined here" note
// and the `#[pin_v2]` suggestion are dropped rather than suppressing the error itself.

extern crate non_pin_project;

use non_pin_project::Foreign;
use std::pin::Pin;

fn project<T>(p: Pin<&mut Foreign<T>>) {
    let &pin mut Foreign(ref pin mut _x) = p;
    //~^ ERROR cannot project on type that is not `#[pin_v2]`
}

fn main() {}
