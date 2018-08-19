// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(const_fn)]

#[derive(PartialEq, Eq)]
enum Cake {
    BlackForest,
    Marmor,
}
use Cake::*;

struct Pair<A, B>(A, B);

const BOO: Pair<Cake, Cake> = Pair(Marmor, BlackForest);
const FOO: Cake = BOO.1;

const fn foo() -> Cake {
    Marmor
}

const WORKS: Cake = Marmor;

const GOO: Cake = foo();

fn main() {
    match BlackForest {
        FOO => println!("hi"),
        GOO => println!("meh"),
        WORKS => println!("möp"),
        _ => println!("bye"),
    }
}
