// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

macro_rules! inner_bind (
    ( $p:pat, $id:ident) => ({let $p = 13; $id}))

macro_rules! outer_bind (
    ($p:pat, $id:ident ) => (inner_bind!($p, $id)))

fn main() {
    outer_bind!(g1,g1);
}

