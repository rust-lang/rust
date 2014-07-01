// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Collection types.
 */

#![experimental]

pub use core_collections::{Collection, Mutable, Map, MutableMap};
pub use core_collections::{Set, MutableSet, Deque};
pub use core_collections::{Bitv, BitvSet, BTree, DList, EnumSet};
pub use core_collections::{PriorityQueue, RingBuf, SmallIntMap};
pub use core_collections::{TreeMap, TreeSet, TrieMap, TrieSet};
pub use core_collections::{bitv, btree, dlist, enum_set};
pub use core_collections::{priority_queue, ringbuf, smallintmap, treemap, trie};

pub use self::hashmap::{HashMap, HashSet};
pub use self::lru_cache::LruCache;

pub mod hashmap;
pub mod lru_cache;
