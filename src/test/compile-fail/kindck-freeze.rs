// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which of the builtin types are considered freezeable.

fn assert_freeze<T:Freeze>() { }
trait Dummy { }

fn test<'a,T,U:Freeze>(_: &'a int) {
    // lifetime pointers are ok...
    assert_freeze::<&'static int>();
    assert_freeze::<&'a int>();
    assert_freeze::<&'a str>();
    assert_freeze::<&'a [int]>();

    // ...unless they are mutable
    assert_freeze::<&'static mut int>(); //~ ERROR does not fulfill `Freeze`
    assert_freeze::<&'a mut int>(); //~ ERROR does not fulfill `Freeze`

    // ~ pointers are ok
    assert_freeze::<~int>();
    assert_freeze::<~str>();
    assert_freeze::<~[int]>();

    // but not if they own a bad thing
    assert_freeze::<~&'a mut int>(); //~ ERROR does not fulfill `Freeze`

    // careful with object types, who knows what they close over...
    assert_freeze::<&'a Dummy>(); //~ ERROR does not fulfill `Freeze`
    assert_freeze::<~Dummy>(); //~ ERROR does not fulfill `Freeze`

    // ...unless they are properly bounded
    assert_freeze::<&'a Dummy:Freeze>();
    assert_freeze::<&'static Dummy:Freeze>();
    assert_freeze::<~Dummy:Freeze>();

    // ...but even then the pointer overrides
    assert_freeze::<&'a mut Dummy:Freeze>(); //~ ERROR does not fulfill `Freeze`

    // closures are like an `&mut` object
    assert_freeze::<&fn()>(); //~ ERROR does not fulfill `Freeze`

    // unsafe ptrs are ok unless they point at unfreezeable things
    assert_freeze::<*int>();
    assert_freeze::<*&'a mut int>(); //~ ERROR does not fulfill `Freeze`
}

fn main() {
}
