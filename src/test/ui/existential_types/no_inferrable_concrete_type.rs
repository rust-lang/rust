// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 52985: Cause cycle error if user code provides no use case that allows an existential type
// to be inferred to a concrete type. This results in an infinite cycle during type normalization.

#![feature(existential_type)]

existential type Foo: Copy; //~ cycle detected

// make compiler happy about using 'Foo'
fn bar(x: Foo) -> Foo { x }

fn main() {
    let _: Foo = std::mem::transmute(0u8);
}
