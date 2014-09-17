// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test related to when a region bound is required to be specified.

trait IsStatic : 'static { }
trait IsSend : Send { }
trait Is<'a> : 'a { }
trait Is2<'a> : 'a { }
trait SomeTrait { }

// Bounds on object types:

struct Foo<'a,'b,'c> {
    // All of these are ok, because we can derive exactly one bound:
    a: Box<IsStatic>,
    b: Box<Is<'static>>,
    c: Box<Is<'a>>,
    d: Box<IsSend>,
    e: Box<Is<'a>+Send>, // we can derive two bounds, but one is 'static, so ok
    f: Box<SomeTrait>, //~ ERROR explicit lifetime bound required
    g: Box<SomeTrait+'a>,

    z: Box<Is<'a>+'b+'c>, //~ ERROR only a single explicit lifetime bound is permitted
}

fn test<
    'a,
    'b,
    A:IsStatic,
    B:Is<'a>+Is2<'b>, // OK in a parameter, but not an object type.
    C:'b+Is<'a>+Is2<'b>,
    D:Is<'a>+Is2<'static>,
    E:'a+'b           // OK in a parameter, but not an object type.
>() { }

fn main() { }
