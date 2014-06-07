// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is an interesting test case. We have a trait (Bar) that is
// implemented for a `Box<Foo>` object (note: no bounds). And then we
// have a `Box<Foo:Send>` object. The impl for `Box<Foo>` is applicable
// to `Box<Foo:Send>` because:
//
// 1. The trait Bar is contravariant w/r/t Self because `Self` appears
//    only in argument position.
// 2. The impl provides `Bar for Box<Foo>`
// 3. The fn `wants_bar()` requires `Bar for Box<Foo:Send>`.
// 4. `Bar for Box<Foo> <: Bar for Box<Foo:Send>` because
//    `Box<Foo:Send> <: Box<Foo>`.

trait Foo { }
struct SFoo;
impl Foo for SFoo { }

trait Bar { fn dummy(&self); }
impl Bar for Box<Foo> { fn dummy(&self) { } }

fn wants_bar<B:Bar>(b: &B) { }

fn main() {
    let x: Box<Foo:Send> = (box SFoo);
    wants_bar(&x);
}


