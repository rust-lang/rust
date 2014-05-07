// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn borrow(_v: &int) {}

fn local() {
    let mut v = box 3;
    borrow(v);
}

fn local_rec() {
    struct F { f: Box<int> }
    let mut v = F {f: box 3};
    borrow(v.f);
}

fn local_recs() {
    struct F { f: G }
    struct G { g: H }
    struct H { h: Box<int> }
    let mut v = F {f: G {g: H {h: box 3}}};
    borrow(v.f.g.h);
}

fn aliased_imm() {
    let mut v = box 3;
    let _w = &v;
    borrow(v);
}

fn aliased_mut() {
    let mut v = box 3;
    let _w = &mut v;
    borrow(v); //~ ERROR cannot borrow `*v`
}

fn aliased_other() {
    let mut v = box 3;
    let mut w = box 4;
    let _x = &mut w;
    borrow(v);
}

fn aliased_other_reassign() {
    let mut v = box 3;
    let mut w = box 4;
    let mut _x = &mut w;
    _x = &mut v;
    borrow(v); //~ ERROR cannot borrow `*v`
}

fn main() {
}
