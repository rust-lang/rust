// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Rec {
    f: ~int,
}

struct Outer {
    f: Inner
}

struct Inner {
    g: Innermost
}

struct Innermost {
    h: ~int,
}

fn borrow(_v: &int) {}

fn box_mut(v: @mut ~int) {
    borrow(*v); //~ ERROR illegal borrow unless pure
}

fn box_mut_rec(v: @mut Rec) {
    borrow(v.f); //~ ERROR illegal borrow unless pure
}

fn box_mut_recs(v: @mut Outer) {
    borrow(v.f.g.h); //~ ERROR illegal borrow unless pure
}

fn box_imm(v: @~int) {
    borrow(*v); // OK
}

fn box_imm_rec(v: @Rec) {
    borrow(v.f); // OK
}

fn box_imm_recs(v: @Outer) {
    borrow(v.f.g.h); // OK
}

fn main() {
}
