// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for what happens when a type parameter `A` is closed over into
// an object. This should yield errors unless `A` (and the object)
// both have suitable bounds.

trait SomeTrait { fn get(&self) -> int; }

fn make_object1<A:SomeTrait>(v: A) -> Box<SomeTrait+'static> {
    box v as Box<SomeTrait+'static>
        //~^ ERROR the parameter type `A` may not live long enough
}

fn make_object2<'a,A:SomeTrait+'a>(v: A) -> Box<SomeTrait+'a> {
    box v as Box<SomeTrait+'a>
}

fn make_object3<'a,'b,A:SomeTrait+'a>(v: A) -> Box<SomeTrait+'b> {
    box v as Box<SomeTrait+'b>
        //~^ ERROR the parameter type `A` may not live long enough
}

fn main() { }
