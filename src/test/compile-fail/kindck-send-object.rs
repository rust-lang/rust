// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which of the builtin types are considered sendable. The tests
// in this file all test the "kind" violates detected during kindck.
// See all `regions-bounded-by-send.rs`

fn assert_send<T:Send>() { }
trait Dummy { }
trait Message : Send { }

// careful with object types, who knows what they close over...

fn object_ref_with_static_bound_not_ok() {
    assert_send::<&'static Dummy+'static>();
    //~^ ERROR the trait `core::kinds::Send` is not implemented
}

fn box_object_with_no_bound_not_ok<'a>() {
    assert_send::<Box<Dummy>>(); //~ ERROR the trait `core::kinds::Send` is not implemented
}

fn proc_with_no_bound_not_ok<'a>() {
    assert_send::<proc()>(); //~ ERROR the trait `core::kinds::Send` is not implemented
}

fn closure_with_no_bound_not_ok<'a>() {
    assert_send::<||:'static>(); //~ ERROR the trait `core::kinds::Send` is not implemented
}

fn object_with_send_bound_ok() {
    assert_send::<&'static Dummy+Send>();
    assert_send::<Box<Dummy+Send>>();
    assert_send::<proc():Send>;
    assert_send::<||:Send>;
}

fn main() { }
