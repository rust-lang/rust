// ignore-tidy-linelength
// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g

// The pretty printers being tested here require the patch from
// https://sourceware.org/bugzilla/show_bug.cgi?id=21763
// min-gdb-version 8.1

// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print btree_set
// gdb-check:$1 = BTreeSet<i32>(len: 15) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

// gdb-command: print btree_map
// gdb-check:$2 = BTreeMap<i32, i32>(len: 15) = {[0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7, [8] = 8, [9] = 9, [10] = 10, [11] = 11, [12] = 12, [13] = 13, [14] = 14}

// gdb-command: print vec_deque
// gdb-check:$3 = VecDeque<i32>(len: 3, cap: 8) = {5, 3, 7}

// gdb-command: print vec_deque2
// gdb-check:$4 = VecDeque<i32>(len: 7, cap: 8) = {2, 3, 4, 5, 6, 7, 8}

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

    // VecDeque where an element was popped.
    let mut vec_deque2 = VecDeque::new();
    for i in 1..8 {
        vec_deque2.push_back(i)
    }
    vec_deque2.pop_front();
    vec_deque2.push_back(8);

    zzz(); // #break
}

fn zzz() { () }
