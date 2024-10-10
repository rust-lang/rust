// See #130494

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

fn f(x: &pin const i32) {}
fn g<'a>(x: &'a pin const i32) {}
fn h<'a>(x: &'a pin mut i32) {}
fn i(x: &pin mut i32) {}
