// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use extra::time;
use extra::treemap::TreeMap;
use std::hashmap::{HashMap, HashSet};
use std::os;
use std::rand::{Rng, IsaacRng, SeedableRng};
use std::trie::TrieMap;
use std::uint;
use std::vec;

fn timed(label: &str, f: ||) {
    let start = time::precise_time_s();
    f();
    let end = time::precise_time_s();
    println!("  {}: {}", label, end - start);
}

fn ascending<M: MutableMap<uint, uint>>(map: &mut M, n_keys: uint) {
    println(" Ascending integers:");

    timed("insert", || {
        for i in range(0u, n_keys) {
            map.insert(i, i + 1);
        }
    });

    timed("search", || {
        for i in range(0u, n_keys) {
            assert_eq!(map.find(&i).unwrap(), &(i + 1));
        }
    });

    timed("remove", || {
        for i in range(0, n_keys) {
            assert!(map.remove(&i));
        }
    });
}

fn descending<M: MutableMap<uint, uint>>(map: &mut M, n_keys: uint) {
    println(" Descending integers:");

    timed("insert", || {
        for i in range(0, n_keys).invert() {
            map.insert(i, i + 1);
        }
    });

    timed("search", || {
        for i in range(0, n_keys).invert() {
            assert_eq!(map.find(&i).unwrap(), &(i + 1));
        }
    });

    timed("remove", || {
        for i in range(0, n_keys) {
            assert!(map.remove(&i));
        }
    });
}

fn vector<M: MutableMap<uint, uint>>(map: &mut M, n_keys: uint, dist: &[uint]) {
    timed("insert", || {
        for i in range(0u, n_keys) {
            map.insert(dist[i], i + 1);
        }
    });

    timed("search", || {
        for i in range(0u, n_keys) {
            assert_eq!(map.find(&dist[i]).unwrap(), &(i + 1));
        }
    });

    timed("remove", || {
        for i in range(0u, n_keys) {
            assert!(map.remove(&dist[i]));
        }
    });
}

fn main() {
    let args = os::args();
    let n_keys = {
        if args.len() == 2 {
            from_str::<uint>(args[1]).unwrap()
        } else {
            1000000
        }
    };

    let mut rand = vec::with_capacity(n_keys);

    {
        let mut rng: IsaacRng = SeedableRng::from_seed(&[1, 1, 1, 1, 1, 1, 1]);
        let mut set = HashSet::new();
        while set.len() != n_keys {
            let next = rng.gen();
            if set.insert(next) {
                rand.push(next);
            }
        }
    }

    println!("{} keys", n_keys);

    println("\nTreeMap:");

    {
        let mut map: TreeMap<uint,uint> = TreeMap::new();
        ascending(&mut map, n_keys);
    }

    {
        let mut map: TreeMap<uint,uint> = TreeMap::new();
        descending(&mut map, n_keys);
    }

    {
        println(" Random integers:");
        let mut map: TreeMap<uint,uint> = TreeMap::new();
        vector(&mut map, n_keys, rand);
    }

    println("\nHashMap:");

    {
        let mut map: HashMap<uint,uint> = HashMap::new();
        ascending(&mut map, n_keys);
    }

    {
        let mut map: HashMap<uint,uint> = HashMap::new();
        descending(&mut map, n_keys);
    }

    {
        println(" Random integers:");
        let mut map: HashMap<uint,uint> = HashMap::new();
        vector(&mut map, n_keys, rand);
    }

    println("\nTrieMap:");

    {
        let mut map: TrieMap<uint> = TrieMap::new();
        ascending(&mut map, n_keys);
    }

    {
        let mut map: TrieMap<uint> = TrieMap::new();
        descending(&mut map, n_keys);
    }

    {
        println(" Random integers:");
        let mut map: TrieMap<uint> = TrieMap::new();
        vector(&mut map, n_keys, rand);
    }
}
