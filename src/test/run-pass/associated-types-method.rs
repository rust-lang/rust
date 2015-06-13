// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that methods whose impl-trait-ref contains associated types
// are supported.

trait Device {
    type Resources;
}
struct Foo<D, R>(D, R);

trait Tr {
    fn present(&self) {}
}

impl<D: Device> Tr for Foo<D, D::Resources> {
    fn present(&self) {}
}

struct Res;
struct Dev;
impl Device for Dev {
    type Resources = Res;
}

fn main() {
    let foo = Foo(Dev, Res);
    foo.present();
}
