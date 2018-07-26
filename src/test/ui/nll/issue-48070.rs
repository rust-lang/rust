// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// revisions: lxl nll

#![cfg_attr(nll, feature(nll))]

struct Foo {
    x: u32
}

impl Foo {
    fn twiddle(&mut self) -> &mut Self { self }
    fn twaddle(&mut self) -> &mut Self { self }
    fn emit(&mut self) {
        self.x += 1;
    }
}

fn main() {
    let mut foo = Foo { x: 0 };
    match 22 {
        22 => &mut foo,
        44 => foo.twiddle(),
        _ => foo.twaddle(),
    }.emit();
}
