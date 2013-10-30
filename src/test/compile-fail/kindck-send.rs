// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which of the builtin types are considered sendable.

fn assert_send<T:Send>() { }
trait Dummy { }

fn test<'a,T,U:Send>(_: &'a int) {
    // lifetime pointers with 'static lifetime are ok
    assert_send::<&'static int>();
    assert_send::<&'static str>();
    assert_send::<&'static [int]>();

    // whether or not they are mutable
    assert_send::<&'static mut int>();

    // otherwise lifetime pointers are not ok
    assert_send::<&'a int>(); //~ ERROR does not fulfill `Send`
    assert_send::<&'a str>(); //~ ERROR does not fulfill `Send`
    assert_send::<&'a [int]>(); //~ ERROR does not fulfill `Send`

    // ~ pointers are ok
    assert_send::<~int>();
    assert_send::<~str>();
    assert_send::<~[int]>();

    // but not if they own a bad thing
    assert_send::<~&'a int>(); //~ ERROR does not fulfill `Send`

    // careful with object types, who knows what they close over...
    assert_send::<&'static Dummy>(); //~ ERROR does not fulfill `Send`
    assert_send::<&'a Dummy>(); //~ ERROR does not fulfill `Send`
    assert_send::<&'a Dummy:Send>(); //~ ERROR does not fulfill `Send`
    assert_send::<~Dummy:>(); //~ ERROR does not fulfill `Send`

    // ...unless they are properly bounded
    assert_send::<&'static Dummy:Send>();
    assert_send::<~Dummy:Send>();

    // but closure and object types can have lifetime bounds which make
    // them not ok (FIXME #5121)
    // assert_send::<~fn:'a()>(); // ERROR does not fulfill `Send`
    // assert_send::<~Dummy:'a>(); // ERROR does not fulfill `Send`

    // unsafe ptrs are ok unless they point at unsendable things
    assert_send::<*int>();
    assert_send::<*&'a int>(); //~ ERROR does not fulfill `Send`
}

fn main() {
}
