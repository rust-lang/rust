// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #24947 ICE using a trait-associated const in an array size


struct Foo;

impl Foo {
    const SIZE: usize = 8;
}

trait Bar {
    const BAR_SIZE: usize;
}

impl Bar for Foo {
    const BAR_SIZE: usize = 12;
}

#[allow(unused_variables)]
fn main() {
    let w: [u8; 12] = [0u8; <Foo as Bar>::BAR_SIZE];
    let x: [u8; 12] = [0u8; <Foo>::BAR_SIZE];
    let y: [u8; 8] = [0u8; <Foo>::SIZE];
    let z: [u8; 8] = [0u8; Foo::SIZE];
}
