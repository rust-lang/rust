// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo { a: int, b: int }

fn main() {
    let mut x = box Foo { a: 1, b: 2 };
    let (a, b) = (&mut x.a, &mut x.b);
    //~^ ERROR cannot borrow `x` (here through borrowing `x.b`) as mutable more than once at a time
    //~^^ NOTE previous borrow of `x` occurs here (through borrowing `x.a`)

    let mut foo = box Foo { a: 1, b: 2 };
    let (c, d) = (&mut foo.a, &foo.b);
    //~^ ERROR cannot borrow `foo` (here through borrowing `foo.b`) as immutable
    //~^^ NOTE previous borrow of `foo` occurs here (through borrowing `foo.a`)
}
