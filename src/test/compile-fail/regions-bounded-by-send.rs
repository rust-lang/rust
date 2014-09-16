// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which of the builtin types are considered sendable. The tests
// in this file all test region bound and lifetime violations that are
// detected during type check.

fn assert_send<T:Send>() { }
trait Dummy { }

// lifetime pointers with 'static lifetime are ok

fn static_lifime_ok<'a,T,U:Send>(_: &'a int) {
    assert_send::<&'static int>();
    assert_send::<&'static str>();
    assert_send::<&'static [int]>();

    // whether or not they are mutable
    assert_send::<&'static mut int>();
}

// otherwise lifetime pointers are not ok

fn param_not_ok<'a>(x: &'a int) {
    assert_send::<&'a int>(); //~ ERROR does not fulfill
}

fn param_not_ok1<'a>(_: &'a int) {
    assert_send::<&'a str>(); //~ ERROR does not fulfill
}

fn param_not_ok2<'a>(_: &'a int) {
    assert_send::<&'a [int]>(); //~ ERROR does not fulfill
}

// boxes are ok

fn box_ok() {
    assert_send::<Box<int>>();
    assert_send::<String>();
    assert_send::<Vec<int>>();
}

// but not if they own a bad thing

fn box_with_region_not_ok<'a>() {
    assert_send::<Box<&'a int>>(); //~ ERROR does not fulfill
}

// objects with insufficient bounds no ok

fn object_with_random_bound_not_ok<'a>() {
    assert_send::<&'a Dummy+'a>(); //~ ERROR does not fulfill
    //~^ ERROR not implemented
}

fn object_with_send_bound_not_ok<'a>() {
    assert_send::<&'a Dummy+Send>(); //~ ERROR does not fulfill
}

fn proc_with_lifetime_not_ok<'a>() {
    assert_send::<proc():'a>(); //~ ERROR does not fulfill
    //~^ ERROR not implemented
}

fn closure_with_lifetime_not_ok<'a>() {
    assert_send::<||:'a>(); //~ ERROR does not fulfill
    //~^ ERROR not implemented
}

// unsafe pointers are ok unless they point at unsendable things

fn unsafe_ok1<'a>(_: &'a int) {
    assert_send::<*const int>();
    assert_send::<*mut int>();
}

fn unsafe_ok2<'a>(_: &'a int) {
    assert_send::<*const &'a int>(); //~ ERROR does not fulfill
}

fn unsafe_ok3<'a>(_: &'a int) {
    assert_send::<*mut &'a int>(); //~ ERROR does not fulfill
}

fn main() {
}
