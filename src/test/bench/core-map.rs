// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use std::oldmap;
use std::treemap::TreeMap;
use core::hashmap::linear::*;
use core::io::WriterUtil;

struct Results {
    sequential_ints: float,
    random_ints: float,
    delete_ints: float,

    sequential_strings: float,
    random_strings: float,
    delete_strings: float
}

fn timed(result: &mut float,
         op: fn()) {
    let start = std::time::precise_time_s();
    op();
    let end = std::time::precise_time_s();
    *result = (end - start);
}

fn old_int_benchmarks(rng: @rand::Rng, num_keys: uint, results: &mut Results) {

    {
        let map = oldmap::HashMap();
        do timed(&mut results.sequential_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(i, i+1);
            }

            for uint::range(0, num_keys) |i| {
                fail_unless!(map.get(&i) == i+1);
            }
        }
    }

    {
        let map = oldmap::HashMap();
        do timed(&mut results.random_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(rng.next() as uint, i);
            }
        }
    }

    {
        let map = oldmap::HashMap();
        for uint::range(0, num_keys) |i| {
            map.insert(i, i);;
        }

        do timed(&mut results.delete_ints) {
            for uint::range(0, num_keys) |i| {
                fail_unless!(map.remove(&i));
            }
        }
    }
}

fn old_str_benchmarks(rng: @rand::Rng, num_keys: uint, results: &mut Results) {
    {
        let map = oldmap::HashMap();
        do timed(&mut results.sequential_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i);
                map.insert(s, i);
            }

            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i);
                fail_unless!(map.get(&s) == i);
            }
        }
    }

    {
        let map = oldmap::HashMap();
        do timed(&mut results.random_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(rng.next() as uint);
                map.insert(s, i);
            }
        }
    }

    {
        let map = oldmap::HashMap();
        for uint::range(0, num_keys) |i| {
            map.insert(uint::to_str(i), i);
        }
        do timed(&mut results.delete_strings) {
            for uint::range(0, num_keys) |i| {
                fail_unless!(map.remove(&uint::to_str(i)));
            }
        }
    }
}

fn linear_int_benchmarks(rng: @rand::Rng, num_keys: uint, results: &mut Results) {
    {
        let mut map = LinearMap::new();
        do timed(&mut results.sequential_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(i, i+1);
            }

            for uint::range(0, num_keys) |i| {
                fail_unless!(map.find(&i).unwrap() == &(i+1));
            }
        }
    }

    {
        let mut map = LinearMap::new();
        do timed(&mut results.random_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(rng.next() as uint, i);
            }
        }
    }

    {
        let mut map = LinearMap::new();
        for uint::range(0, num_keys) |i| {
            map.insert(i, i);;
        }

        do timed(&mut results.delete_ints) {
            for uint::range(0, num_keys) |i| {
                fail_unless!(map.remove(&i));
            }
        }
    }
}

fn linear_str_benchmarks(rng: @rand::Rng, num_keys: uint, results: &mut Results) {
    {
        let mut map = LinearMap::new();
        do timed(&mut results.sequential_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i);
                map.insert(s, i);
            }

            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i);
                fail_unless!(map.find(&s).unwrap() == &i);
            }
        }
    }

    {
        let mut map = LinearMap::new();
        do timed(&mut results.random_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(rng.next() as uint);
                map.insert(s, i);
            }
        }
    }

    {
        let mut map = LinearMap::new();
        for uint::range(0, num_keys) |i| {
            map.insert(uint::to_str(i), i);
        }
        do timed(&mut results.delete_strings) {
            for uint::range(0, num_keys) |i| {
                fail_unless!(map.remove(&uint::to_str(i)));
            }
        }
    }
}

fn tree_int_benchmarks(rng: @rand::Rng, num_keys: uint, results: &mut Results) {
    {
        let mut map = TreeMap::new();
        do timed(&mut results.sequential_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(i, i+1);
            }

            for uint::range(0, num_keys) |i| {
                fail_unless!(map.find(&i).unwrap() == &(i+1));
            }
        }
    }

    {
        let mut map = TreeMap::new();
        do timed(&mut results.random_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(rng.next() as uint, i);
            }
        }
    }

    {
        let mut map = TreeMap::new();
        for uint::range(0, num_keys) |i| {
            map.insert(i, i);;
        }

        do timed(&mut results.delete_ints) {
            for uint::range(0, num_keys) |i| {
                fail_unless!(map.remove(&i));
            }
        }
    }
}

fn tree_str_benchmarks(rng: @rand::Rng, num_keys: uint, results: &mut Results) {
    {
        let mut map = TreeMap::new();
        do timed(&mut results.sequential_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i);
                map.insert(s, i);
            }

            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i);
                fail_unless!(map.find(&s).unwrap() == &i);
            }
        }
    }

    {
        let mut map = TreeMap::new();
        do timed(&mut results.random_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(rng.next() as uint);
                map.insert(s, i);
            }
        }
    }

    {
        let mut map = TreeMap::new();
        for uint::range(0, num_keys) |i| {
            map.insert(uint::to_str(i), i);
        }
        do timed(&mut results.delete_strings) {
            for uint::range(0, num_keys) |i| {
                fail_unless!(map.remove(&uint::to_str(i)));
            }
        }
    }
}

fn write_header(header: &str) {
    io::stdout().write_str(header);
    io::stdout().write_str("\n");
}

fn write_row(label: &str, value: float) {
    io::stdout().write_str(fmt!("%30s %f s\n", label, value));
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
        sequential_ints: 0f,
        random_ints: 0f,
        delete_ints: 0f,

        sequential_strings: 0f,
        random_strings: 0f,
        delete_strings: 0f,
    }
}

fn main() {
    let args = os::args();
    let num_keys = {
        if args.len() == 2 {
            uint::from_str(args[1]).get()
        } else {
            100 // woefully inadequate for any real measurement
        }
    };

    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    {
        let rng = rand::seeded_rng(seed);
        let mut results = empty_results();
        old_int_benchmarks(rng, num_keys, &mut results);
        old_str_benchmarks(rng, num_keys, &mut results);
        write_results("std::oldmap::HashMap", &results);
    }

    {
        let rng = rand::seeded_rng(seed);
        let mut results = empty_results();
        linear_int_benchmarks(rng, num_keys, &mut results);
        linear_str_benchmarks(rng, num_keys, &mut results);
        write_results("core::hashmap::linear::LinearMap", &results);
    }

    {
        let rng = rand::seeded_rng(seed);
        let mut results = empty_results();
        tree_int_benchmarks(rng, num_keys, &mut results);
        tree_str_benchmarks(rng, num_keys, &mut results);
        write_results("std::treemap::TreeMap", &results);
    }
}
