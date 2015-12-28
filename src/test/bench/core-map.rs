// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(std_misc, rand, time2)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::__rand::{Rng, thread_rng};
use std::time::Instant;

fn timed<F>(label: &str, mut f: F) where F: FnMut() {
    let start = Instant::now();
    f();
    println!("  {}: {:?}", label, start.elapsed());
}

trait MutableMap {
    fn insert(&mut self, k: usize, v: usize);
    fn remove(&mut self, k: &usize) -> bool;
    fn find(&self, k: &usize) -> Option<&usize>;
}

impl MutableMap for BTreeMap<usize, usize> {
    fn insert(&mut self, k: usize, v: usize) { self.insert(k, v); }
    fn remove(&mut self, k: &usize) -> bool { self.remove(k).is_some() }
    fn find(&self, k: &usize) -> Option<&usize> { self.get(k) }
}
impl MutableMap for HashMap<usize, usize> {
    fn insert(&mut self, k: usize, v: usize) { self.insert(k, v); }
    fn remove(&mut self, k: &usize) -> bool { self.remove(k).is_some() }
    fn find(&self, k: &usize) -> Option<&usize> { self.get(k) }
}

fn ascending<M: MutableMap>(map: &mut M, n_keys: usize) {
    println!(" Ascending integers:");

    timed("insert", || {
        for i in 0..n_keys {
            map.insert(i, i + 1);
        }
    });

    timed("search", || {
        for i in 0..n_keys {
            assert_eq!(map.find(&i).unwrap(), &(i + 1));
        }
    });

    timed("remove", || {
        for i in 0..n_keys {
            assert!(map.remove(&i));
        }
    });
}

fn descending<M: MutableMap>(map: &mut M, n_keys: usize) {
    println!(" Descending integers:");

    timed("insert", || {
        for i in (0..n_keys).rev() {
            map.insert(i, i + 1);
        }
    });

    timed("search", || {
        for i in (0..n_keys).rev() {
            assert_eq!(map.find(&i).unwrap(), &(i + 1));
        }
    });

    timed("remove", || {
        for i in 0..n_keys {
            assert!(map.remove(&i));
        }
    });
}

fn vector<M: MutableMap>(map: &mut M, n_keys: usize, dist: &[usize]) {
    timed("insert", || {
        for i in 0..n_keys {
            map.insert(dist[i], i + 1);
        }
    });

    timed("search", || {
        for i in 0..n_keys {
            assert_eq!(map.find(&dist[i]).unwrap(), &(i + 1));
        }
    });

    timed("remove", || {
        for i in 0..n_keys {
            assert!(map.remove(&dist[i]));
        }
    });
}

fn main() {
    let mut args = env::args();
    let n_keys = {
        if args.len() == 2 {
            args.nth(1).unwrap().parse::<usize>().unwrap()
        } else {
            1000000
        }
    };

    let mut rand = Vec::with_capacity(n_keys);

    {
        let seed: &[_] = &[1, 1, 1, 1, 1, 1, 1];
        let mut rng = thread_rng();
        let mut set = HashSet::new();
        while set.len() != n_keys {
            let next = rng.gen();
            if set.insert(next) {
                rand.push(next);
            }
        }
    }

    println!("{} keys", n_keys);

    println!("\nBTreeMap:");

    {
        let mut map: BTreeMap<usize,usize> = BTreeMap::new();
        ascending(&mut map, n_keys);
    }

    {
        let mut map: BTreeMap<usize,usize> = BTreeMap::new();
        descending(&mut map, n_keys);
    }

    {
        println!(" Random integers:");
        let mut map: BTreeMap<usize,usize> = BTreeMap::new();
        vector(&mut map, n_keys, &rand);
    }

    println!("\nHashMap:");

    {
        let mut map: HashMap<usize,usize> = HashMap::new();
        ascending(&mut map, n_keys);
    }

    {
        let mut map: HashMap<usize,usize> = HashMap::new();
        descending(&mut map, n_keys);
    }

    {
        println!(" Random integers:");
        let mut map: HashMap<usize,usize> = HashMap::new();
        vector(&mut map, n_keys, &rand);
    }
}
