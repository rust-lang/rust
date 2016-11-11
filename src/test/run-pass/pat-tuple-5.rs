// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn tuple() {
    struct S;
    struct Z;
    struct W;
    let x = (S, Z, W);
    match x { (S, ..) => {} }
    match x { (.., W) => {} }
    match x { (S, .., W) => {} }
    match x { (.., Z, _) => {} }
}

fn tuple_struct() {
    struct SS(S, Z, W);

    struct S;
    struct Z;
    struct W;
    let x = SS(S, Z, W);
    match x { SS(S, ..) => {} }
    match x { SS(.., W) => {} }
    match x { SS(S, .., W) => {} }
    match x { SS(.., Z, _) => {} }
}

fn main() {
    tuple();
    tuple_struct();
}
