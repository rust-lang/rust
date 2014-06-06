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

extern crate collections;
extern crate rand;
extern crate time;

use std::collections::bitv::BitvSet;
use std::collections::TreeSet;
use std::collections::HashSet;
use std::os;
use std::uint;

struct Results {
    sequential_ints: f64,
    random_ints: f64,
    delete_ints: f64,

    sequential_strings: f64,
    random_strings: f64,
    delete_strings: f64
}

fn timed(result: &mut f64, op: ||) {
    let start = time::precise_time_s();
    op();
    let end = time::precise_time_s();
    *result = (end - start);
}

impl Results {
    pub fn bench_int<T:MutableSet<uint>,
                     R: rand::Rng>(
                     &mut self,
                     rng: &mut R,
                     num_keys: uint,
                     rand_cap: uint,
                     f: || -> T) { {
            let mut set = f();
            timed(&mut self.sequential_ints, || {
                for i in range(0u, num_keys) {
                    set.insert(i);
                }

                for i in range(0u, num_keys) {
                    assert!(set.contains(&i));
                }
            })
        }

        {
            let mut set = f();
            timed(&mut self.random_ints, || {
                for _ in range(0, num_keys) {
                    set.insert(rng.gen::<uint>() % rand_cap);
                }
            })
        }

        {
            let mut set = f();
            for i in range(0u, num_keys) {
                set.insert(i);
            }

            timed(&mut self.delete_ints, || {
                for i in range(0u, num_keys) {
                    assert!(set.remove(&i));
                }
            })
        }
    }

    pub fn bench_str<T:MutableSet<String>,
                     R:rand::Rng>(
                     &mut self,
                     rng: &mut R,
                     num_keys: uint,
                     f: || -> T) {
        {
            let mut set = f();
            timed(&mut self.sequential_strings, || {
                for i in range(0u, num_keys) {
                    set.insert(i.to_str());
                }

                for i in range(0u, num_keys) {
                    assert!(set.contains(&i.to_str()));
                }
            })
        }

        {
            let mut set = f();
            timed(&mut self.random_strings, || {
                for _ in range(0, num_keys) {
                    let s = rng.gen::<uint>().to_str();
                    set.insert(s);
                }
            })
        }

        {
            let mut set = f();
            for i in range(0u, num_keys) {
                set.insert(i.to_str());
            }
            timed(&mut self.delete_strings, || {
                for i in range(0u, num_keys) {
                    assert!(set.remove(&i.to_str()));
                }
            })
        }
    }
}

fn write_header(header: &str) {
    println!("{}", header);
}

fn write_row(label: &str, value: f64) {
    println!("{:30s} {} s\n", label, value);
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
        sequential_ints: 0.0,
        random_ints: 0.0,
        delete_ints: 0.0,

        sequential_strings: 0.0,
        random_strings: 0.0,
        delete_strings: 0.0,
    }
}

fn main() {
    let args = os::args();
    let args = args.as_slice();
    let num_keys = {
        if args.len() == 2 {
            from_str::<uint>(args[1].as_slice()).unwrap()
        } else {
            100 // woefully inadequate for any real measurement
        }
    };

    let seed = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let max = 200000;

    {
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);
        let mut results = empty_results();
        results.bench_int(&mut rng, num_keys, max, || {
            let s: HashSet<uint> = HashSet::new();
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
            let s: TreeSet<uint> = TreeSet::new();
            s
        });
        results.bench_str(&mut rng, num_keys, || {
            let s: TreeSet<String> = TreeSet::new();
            s
        });
        write_results("collections::TreeSet", &results);
    }

    {
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);
        let mut results = empty_results();
        results.bench_int(&mut rng, num_keys, max, || BitvSet::new());
        write_results("collections::bitv::BitvSet", &results);
    }
}
