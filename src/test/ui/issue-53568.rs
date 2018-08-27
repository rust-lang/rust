// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for an NLL-related ICE (#53568) -- we failed to
// resolve inference variables in "custom type-ops".
//
// compile-pass

#![feature(nll)]
#![allow(dead_code)]

trait Future {
    type Item;
}

impl<F, T> Future for F
where F: Fn() -> T
{
    type Item = T;
}

trait Connect {}

struct Connector<H> {
    handler: H,
}

impl<H, T> Connect for Connector<H>
where
    T: 'static,
    H: Future<Item = T>
{
}

struct Client<C> {
    connector: C,
}

fn build<C>(_connector: C) -> Client<C> {
    unimplemented!()
}

fn client<H>(handler: H) -> Client<impl Connect>
where H: Fn() + Copy
{
    let connector = Connector {
        handler,
    };
    let client = build(connector);
    client
}

fn main() { }

