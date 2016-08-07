// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    foo: Vec<u32>,
}

impl Copy for Foo { }
//~^ ERROR E0204
//~| NOTE field `foo` does not implement `Copy`

#[derive(Copy)]
//~^ ERROR E0204
//~| NOTE field `ty` does not implement `Copy`
//~| NOTE in this expansion of #[derive(Copy)]
struct Foo2<'a> {
    ty: &'a mut bool,
}

fn main() {
}
