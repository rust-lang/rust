// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// When mutably indexing a type that implements `Index` and `IndexMut` but
// `Index::index` is being used specifically, the normal special help message
// should not mention a missing `IndexMut` impl.

fn main() {
    use std::ops::Index;

    let v = String::from("dinosaur");
    Index::index(&v, 1..2).make_ascii_uppercase(); //~ ERROR
}
