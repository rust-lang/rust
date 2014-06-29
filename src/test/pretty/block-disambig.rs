// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A bunch of tests for syntactic forms involving blocks that were
// previously ambiguous (e.g. 'if true { } *val;' gets parsed as a
// binop)

#![feature(managed_boxes)]

use std::cell::Cell;
use std::gc::GC;

fn test1() { let val = box(GC) 0i; { } *val; }

fn test2() -> int { let val = box(GC) 0i; { } *val }

struct S { eax: int }

fn test3() {
    let regs = box(GC) Cell::new(S {eax: 0});
    match true { true => { } _ => { } }
    regs.set(S {eax: 1});
}

fn test4() -> bool { let regs = box(GC) true; if true { } *regs || false }

fn test5() -> (int, int) { { } (0, 1) }

fn test6() -> bool { { } (true || false) && true }

fn test7() -> uint {
    let regs = box(GC) 0i;
    match true { true => { } _ => { } }
    (*regs < 2) as uint
}

fn test8() -> int {
    let val = box(GC) 0i;
    match true {
        true => { }
        _    => { }
    }
    if *val < 1 {
        0
    } else {
        1
    }
}

fn test9() {
    let regs = box(GC) Cell::new(0i);
    match true { true => { } _ => { } } regs.set(regs.get() + 1);
}

fn test10() -> int {
    let regs = box(GC) vec!(0i);
    match true { true => { } _ => { } }
    *(*regs).get(0)
}

fn test11() -> Vec<int> { if true { } vec!(1, 2) }
