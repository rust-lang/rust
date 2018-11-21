// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print btree_set
// gdb-check:$1 = BTreeSet<i32>(len: 15) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

// gdb-command: print btree_map
// gdb-check:$2 = BTreeMap<i32, i32>(len: 15) = {[0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7, [8] = 8, [9] = 9, [10] = 10, [11] = 11, [12] = 12, [13] = 13, [14] = 14}

// gdb-command: print vec_deque
// gdb-check:$3 = VecDeque<i32>(len: 3, cap: 8) = {5, 3, 7}

#![allow(unused_variables)]
use std::collections::BTreeSet;
use std::collections::BTreeMap;
use std::collections::VecDeque;


fn main() {

    // BTreeSet
    let mut btree_set = BTreeSet::new();
    for i in 0..15 {
        btree_set.insert(i);
    }

    // BTreeMap
    let mut btree_map = BTreeMap::new();
    for i in 0..15 {
        btree_map.insert(i, i);
    }

    // VecDeque
    let mut vec_deque = VecDeque::new();
    vec_deque.push_back(5);
    vec_deque.push_back(3);
    vec_deque.push_back(7);

    zzz(); // #break
}

fn zzz() { () }
