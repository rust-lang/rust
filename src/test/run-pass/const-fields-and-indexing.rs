// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const x : [int * 4] = [1,2,3,4];
const p : int = x[2];
const y : &[int] = &[1,2,3,4];
const q : int = y[2];

const s : {a: int, b: int} = {a: 10, b: 20};
const t : int = s.b;

const k : {a: int, b: int, c: {d: int, e: int}} = {a: 10, b: 20, c: {d: 30,
                                                                     e: 40}};
const m : int = k.c.e;

fn main() {
    io::println(fmt!("%?", p));
    io::println(fmt!("%?", q));
    io::println(fmt!("%?", t));
    assert p == 3;
    assert q == 3;
    assert t == 20;
}
