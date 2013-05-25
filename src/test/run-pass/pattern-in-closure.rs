// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    x: int,
    y: int
}

pub fn main() {
    let f = |(x, _): (int, int)| println((x + 1).to_str());
    let g = |Foo { x: x, y: y }: Foo| println((x + 1).to_str());
    f((2, 3));
    g(Foo { x: 1, y: 2 });
}
