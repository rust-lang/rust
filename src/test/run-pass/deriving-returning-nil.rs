// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Show {
    #[derivable]
    fn show(&self);
}

impl int : Show {
    fn show(&self) {
        io::println(self.to_str());
    }
}

struct Foo {
    x: int,
    y: int,
    z: int,
}

impl Foo : Show;

enum Bar {
    Baz(int, int),
    Boo(Foo),
}

impl Bar : Show;

fn main() {
    let foo = Foo { x: 1, y: 2, z: 3 };
    foo.show();

    io::println("---");

    let baz = Baz(4, 5);
    baz.show();

    io::println("---");

    let boo = Boo(Foo { x: 6, y: 7, z: 8 });
    boo.show();
}

