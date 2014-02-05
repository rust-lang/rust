// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    f: int
}

impl Foo {
    fn bar(&self) -> int {
        println!("f is {}", self.f);
        self.f
    }
}

trait Tr {
    fn tar(&self, x: Box<Foo>) -> Foo;
}

impl Tr for Foo {
    fn tar(&self, x: Box<Foo>) -> Foo {
        Foo{ f: self.f + x.f }
    }
}

trait Tr2<X, Y: Tr> {
    fn squid(&self, y: &Y, z: Self) -> Box<X>;
}

impl Tr2<Foo, Foo> for Foo {
    fn squid(&self, y: &Foo, z: Foo) -> Box<Foo> {
        box Foo { f: y.f + z.f + self.f }
    }
}

enum En {
    Var1,
    Var2,
    Var3(int, int, Foo)
}

fn main() {
    let x = Foo { f: 237 };
    let _f = x.bar();
    let en = Var2;

    let _ = match en {
        Var1 => x.bar(),
        Var2 => 34,
        Var3(x, y, f) => f.bar()
    };
}
