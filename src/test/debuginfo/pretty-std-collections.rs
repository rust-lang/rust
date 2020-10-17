// ignore-tidy-linelength
// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g

// The pretty printers being tested here require the patch from
// https://sourceware.org/bugzilla/show_bug.cgi?id=21763
// min-gdb-version: 8.1

// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print btree_set
// gdb-check:$1 = BTreeSet(size=15) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

// gdb-command: print empty_btree_set
// gdb-check:$2 = BTreeSet(size=0)

// gdb-command: print btree_map
// gdb-check:$3 = BTreeMap(size=15) = {[0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7, [8] = 8, [9] = 9, [10] = 10, [11] = 11, [12] = 12, [13] = 13, [14] = 14}

// gdb-command: print empty_btree_map
// gdb-check:$4 = BTreeMap(size=0)

// gdb-command: print option_btree_map
// gdb-check:$5 = BTreeMap(size=2) = {[false] = [...], [true] = [...]}
// (abbreviated because both values vary wildly over gdb versions and/or linux distributions)

// gdb-command: print nasty_btree_map
// gdb-check:$6 = BTreeMap(size=15) = {[0] = pretty_std_collections::MyLeafNode (0), [...]}
// (abbreviated because it's boring but we need enough elements to include internal nodes)

// gdb-command: print zst_key_btree_map
// gdb-check:$7 = BTreeMap(size=1) = {[()] = 1}

// gdb-command: print zst_val_btree_map
// gdb-check:$8 = BTreeMap(size=1) = {[1] = ()}

// gdb-command: print zst_key_val_btree_map
// gdb-check:$9 = BTreeMap(size=1) = {[()] = ()}

// gdb-command: print vec_deque
// gdb-check:$10 = VecDeque(size=3) = {5, 3, 7}

// gdb-command: print vec_deque2
// gdb-check:$11 = VecDeque(size=7) = {2, 3, 4, 5, 6, 7, 8}

// gdb-command: print hash_map
// gdb-check:$12 = HashMap(size=4) = {[1] = 10, [2] = 20, [3] = 30, [4] = 40}

// gdb-command: print hash_set
// gdb-check:$13 = HashSet(size=4) = {1, 2, 3, 4}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print vec_deque
// lldbg-check:[...]$0 = size=3 { [0] = 5 [1] = 3 [2] = 7 }
// lldbr-check:(alloc::collections::vec_deque::VecDeque<i32>) vec_deque = size=3 = { [0] = 5 [1] = 3 [2] = 7 }

// lldb-command:print vec_deque2
// lldbg-check:[...]$1 = size=7 { [0] = 2 [1] = 3 [2] = 4 [3] = 5 [4] = 6 [5] = 7 [6] = 8 }
// lldbr-check:(alloc::collections::vec_deque::VecDeque<i32>) vec_deque2 = size=7 = { [0] = 2 [1] = 3 [2] = 4 [3] = 5 [4] = 6 [5] = 7 [6] = 8 }

// lldb-command:print hash_map
// lldbg-check:[...]$2 = size=4 { [0] = { 0 = 1 1 = 10 } [1] = { 0 = 2 1 = 20 } [2] = { 0 = 3 1 = 30 } [3] = { 0 = 4 1 = 40 } }
// lldbr-check:(std::collections::hash::map::HashMap<u64, u64, [...]>) hash_map = size=4 size=4 { [0] = { 0 = 1 1 = 10 } [1] = { 0 = 2 1 = 20 } [2] = { 0 = 3 1 = 30 } [3] = { 0 = 4 1 = 40 } }

// lldb-command:print hash_set
// lldbg-check:[...]$3 = size=4 { [0] = 1 [1] = 2 [2] = 3 [3] = 4 }
// lldbr-check:(std::collections::hash::set::HashSet<u64, [...]>) hash_set = size=4 { [0] = 1 [1] = 2 [2] = 3 [3] = 4 }

#![allow(unused_variables)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::{BuildHasherDefault, Hasher};

struct MyLeafNode(i32); // helps to ensure we don't blindly replace substring "LeafNode"

#[derive(Default)]
struct SimpleHasher { hash: u64 }

impl Hasher for SimpleHasher {
    fn finish(&self) -> u64 { self.hash }
    fn write(&mut self, bytes: &[u8]) {}
    fn write_u64(&mut self, i: u64) { self.hash = i }
}

fn main() {
    // BTreeSet
    let mut btree_set = BTreeSet::new();
    for i in 0..15 {
        btree_set.insert(i);
    }

    let mut empty_btree_set: BTreeSet<i32> = BTreeSet::new();

    // BTreeMap
    let mut btree_map = BTreeMap::new();
    for i in 0..15 {
        btree_map.insert(i, i);
    }

    let mut empty_btree_map: BTreeMap<i32, u32> = BTreeMap::new();

    let mut option_btree_map: BTreeMap<bool, Option<bool>> = BTreeMap::new();
    option_btree_map.insert(false, None);
    option_btree_map.insert(true, Some(true));

    let mut nasty_btree_map: BTreeMap<i32, MyLeafNode> = BTreeMap::new();
    for i in 0..15 {
        nasty_btree_map.insert(i, MyLeafNode(i));
    }

    let mut zst_key_btree_map: BTreeMap<(), i32> = BTreeMap::new();
    zst_key_btree_map.insert((), 1);

    let mut zst_val_btree_map: BTreeMap<i32, ()> = BTreeMap::new();
    zst_val_btree_map.insert(1, ());

    let mut zst_key_val_btree_map: BTreeMap<(), ()> = BTreeMap::new();
    zst_key_val_btree_map.insert((), ());

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

    // HashMap
    let mut hash_map = HashMap::<u64, u64, BuildHasherDefault<SimpleHasher>>::default();
    for i in 1..5 {
        hash_map.insert(i, i * 10);
    }

    // HashSet
    let mut hash_set = HashSet::<u64, BuildHasherDefault<SimpleHasher>>::default();
    for i in 1..5 {
        hash_set.insert(i);
    }

    zzz(); // #break
}

fn zzz() {
    ()
}
