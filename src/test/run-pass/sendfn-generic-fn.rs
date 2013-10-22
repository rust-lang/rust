// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::task;

pub fn main() { test05(); }

#[deriving(Clone)]
struct Pair<A,B> {
    a: A,
    b: B,
}

fn make_generic_record<A,B>(a: A, b: B) -> Pair<A,B> {
    return Pair {a: a, b: b};
}

fn test05_start(f: &~fn(v: f64, v: ~str) -> Pair<f64, ~str>) {
    let p = (*f)(22.22, ~"Hi");
    info!("{:?}", p.clone());
    assert!(p.a == 22.22);
    assert!(p.b == ~"Hi");

    let q = (*f)(44.44, ~"Ho");
    info!("{:?}", q.clone());
    assert!(q.a == 44.44);
    assert!(q.b == ~"Ho");
}

fn spawn<A,B>(f: extern fn(&~fn(A,B)->Pair<A,B>)) {
    let arg: ~fn(A, B) -> Pair<A,B> = |a, b| make_generic_record(a, b);
    task::spawn(|| f(&arg));
}

fn test05() {
    spawn::<f64,~str>(test05_start);
}
