// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This code was creating an ICE in the MIR type checker. The reason
// is that we are reifying a reference to a function (`foo::<'x>`),
// which involves extracting its signature, but we were not
// normalizing the signature afterwards. As a result, we sometimes got
// errors around the `<u32 as Foo<'x>>::Value`, which can be
// normalized to `f64`.

#![allow(dead_code)]

trait Foo<'x> {
    type Value;
}

impl<'x> Foo<'x> for u32 {
    type Value = f64;
}

struct Providers<'x> {
    foo: for<'y> fn(x: &'x u32, y: &'y u32) -> <u32 as Foo<'x>>::Value,
}

fn foo<'y, 'x: 'x>(x: &'x u32, y: &'y u32) -> <u32 as Foo<'x>>::Value {
    *x as f64
}

fn main() {
    Providers { foo };
}
