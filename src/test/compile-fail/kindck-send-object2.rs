// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Continue kindck-send-object1.rs.

fn assert_send<T:Send>() { }
trait Dummy { }

fn test50() {
    assert_send::<&'static Dummy>(); //~ ERROR does not fulfill `Send`
}

fn test53() {
    assert_send::<Box<Dummy>>(); //~ ERROR does not fulfill `Send`
}

// ...unless they are properly bounded
fn test60() {
    assert_send::<&'static Dummy+Send>();
}
fn test61() {
    assert_send::<Box<Dummy+Send>>();
}

fn main() { }
