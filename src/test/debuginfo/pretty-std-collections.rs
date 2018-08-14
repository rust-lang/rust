// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print btree_set
// gdb-check:$1 = BTreeSet<i32>(len: 3) = {3, 5, 7}

// gdb-command: print vec_deque
// gdb-check:$2 = VecDeque<i32>(len: 3, cap: 8) = {5, 3, 7}

#![allow(unused_variables)]
use std::collections::BTreeSet;
use std::collections::VecDeque;


fn main() {

    // BTreeSet
    let mut btree_set = BTreeSet::new();
    btree_set.insert(5);
    btree_set.insert(3);
    btree_set.insert(7);

    // VecDeque
    let mut vec_deque = VecDeque::new();
    vec_deque.push_back(5);
    vec_deque.push_back(3);
    vec_deque.push_back(7);

    zzz(); // #break
}

fn zzz() { () }
