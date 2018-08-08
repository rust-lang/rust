// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll
// revisions: ast mir
//[mir]compile-flags: -Z borrowck=compare

struct TrieMapIterator<'a> {
    node: &'a usize
}

fn main() {
    let a = 5;
    let _iter = TrieMapIterator{node: &a};
    _iter.node = & //[ast]~ ERROR cannot assign to field `_iter.node` of immutable binding
                   //[mir]~^ ERROR cannot assign to field `_iter.node` of immutable binding (Ast)
                   // MIR doesn't generate an error because the code isn't reachable. This is OK
                   // because the test is here to check that the compiler doesn't ICE (cf. #5500).
    panic!()
}
