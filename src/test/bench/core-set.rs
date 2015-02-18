// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty very bad with line comments

#![feature(unboxed_closures)]

extern crate collections;
extern crate rand;

use std::collections::BTreeSet;
use std::collections::BitSet;
use std::collections::HashSet;
use std::hash::Hash;
use std::env;
use std::time::Duration;

struct Results {
    sequential_ints: Duration,
    random_ints: Duration,
    delete_ints: Duration,

    sequential_strings: Duration,
    random_strings: Duration,
    delete_strings: Duration,
}

fn timed<F>(result: &mut Duration, op: F) where F: FnOnce() {
    *result = Duration::span(op);
}

trait MutableSet<T> {
    fn insert(&mut self, k: T);
    fn remove(&mut self, k: &T) -> bool;
    fn contains(&self, k: &T) -> bool;
}

impl<T: Hash + Eq> MutableSet<T> for HashSet<T> {
    fn insert(&mut self, k: T) { self.insert(k); }
    fn remove(&mut self, k: &T) -> bool { self.remove(k) }
    fn contains(&self, k: &T) -> bool { self.contains(k) }
}
impl<T: Ord> MutableSet<T> for BTreeSet<T> {
    fn insert(&mut self, k: T) { self.insert(k); }
    fn remove(&mut self, k: &T) -> bool { self.remove(k) }
    fn contains(&self, k: &T) -> bool { self.contains(k) }
}
impl MutableSet<usize> for BitSet {
    fn insert(&mut self, k: usize) { self.insert(k); }
    fn remove(&mut self, k: &usize) -> bool { self.remove(k) }
    fn contains(&self, k: &usize) -> bool { self.contains(k) }
}

impl Results {
    pub fn bench_int<T:MutableSet<usize>,
                     R:rand::Rng,
                     F:FnMut() -> T>(
                     &mut self,
                     rng: &mut R,
                     num_keys: usize,
                     rand_cap: usize,
                     mut f: F) {
        {
            let mut set = f();
            timed(&mut self.sequential_ints, || {
                for i in 0..num_keys {
                    set.insert(i);
                }

                for i in 0..num_keys {
                    assert!(set.contains(&i));
                }
            })
        }

        {
            let mut set = f();
            timed(&mut self.random_ints, || {
                for _ in 0..num_keys {
                    set.insert(rng.gen::<usize>() % rand_cap);
                }
            })
        }

        {
            let mut set = f();
            for i in 0..num_keys {
                set.insert(i);
            }

            timed(&mut self.delete_ints, || {
                for i in 0..num_keys {
                    assert!(set.remove(&i));
                }
            })
        }
    }

    pub fn bench_str<T:MutableSet<String>,
                     R:rand::Rng,
                     F:FnMut() -> T>(
                     &mut self,
                     rng: &mut R,
                     num_keys: usize,
                     mut f: F) {
        {
            let mut set = f();
            timed(&mut self.sequential_strings, || {
                for i in 0..num_keys {
                    set.insert(i.to_string());
                }

                for i in 0..num_keys {
                    assert!(set.contains(&i.to_string()));
                }
            })
        }

        {
            let mut set = f();
            timed(&mut self.random_strings, || {
                for _ in 0..num_keys {
                    let s = rng.gen::<usize>().to_string();
                    set.insert(s);
                }
            })
        }

        {
            let mut set = f();
            for i in 0..num_keys {
                set.insert(i.to_string());
            }
            timed(&mut self.delete_strings, || {
                for i in 0..num_keys {
                    assert!(set.remove(&i.to_string()));
                }
            })
        }
    }
}

fn write_header(header: &str) {
    println!("{}", header);
}

fn write_row(label: &str, value: Duration) {
    println!("{:30} {} s\n", label, value);
}

fn write_results(label: &str, results: &Results) {
    write_header(label);
    write_row("sequential_ints", results.sequential_ints);
    write_row("random_ints", results.random_ints);
    write_row("delete_ints", results.delete_ints);
    write_row("sequential_strings", results.sequential_strings);
    write_row("random_strings", results.random_strings);
    write_row("delete_strings", results.delete_strings);
}

fn empty_results() -> Results {
    Results {
        sequential_ints: Duration::seconds(0),
        random_ints: Duration::seconds(0),
        delete_ints: Duration::seconds(0),

        sequential_strings: Duration::seconds(0),
        random_strings: Duration::seconds(0),
        delete_strings: Duration::seconds(0),
    }
}

fn main() {
    let mut args = env::args();
    let num_keys = {
        if args.len() == 2 {
            args.nth(1).unwrap().parse::<usize>().unwrap()
        } else {
            100 // woefully inadequate for any real measurement
        }
    };

    let seed: &[_] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let max = 200000;

    {
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);
        let mut results = empty_results();
        results.bench_int(&mut rng, num_keys, max, || {
            let s: HashSet<usize> = HashSet::new();
            s
        });
        results.bench_str(&mut rng, num_keys, || {
            let s: HashSet<String> = HashSet::new();
            s
        });
        write_results("collections::HashSet", &results);
    }

    {
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);
        let mut results = empty_results();
        results.bench_int(&mut rng, num_keys, max, || {
            let s: BTreeSet<usize> = BTreeSet::new();
            s
        });
        results.bench_str(&mut rng, num_keys, || {
            let s: BTreeSet<String> = BTreeSet::new();
            s
        });
        write_results("collections::BTreeSet", &results);
    }

    {
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);
        let mut results = empty_results();
        results.bench_int(&mut rng, num_keys, max, || BitSet::new());
        write_results("collections::bit_vec::BitSet", &results);
    }
}
