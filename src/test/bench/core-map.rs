/*



*/

use std;
use std::map;
use mutable::Mut;
use send_map::linear::*;
use io::WriterUtil;

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

fn int_benchmarks<M: map::Map<uint, uint>>(make_map: fn() -> M,
                                           rng: @rand::Rng,
                                           num_keys: uint,
                                           results: &mut Results) {

    {
        let map = make_map();
        do timed(&mut results.sequential_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(i, i+1);
            }

            for uint::range(0, num_keys) |i| {
                assert map.get(i) == i+1;
            }
        }
    }

    {
        let map = make_map();
        do timed(&mut results.random_ints) {
            for uint::range(0, num_keys) |i| {
                map.insert(rng.next() as uint, i);
            }
        }
    }

    {
        let map = make_map();
        for uint::range(0, num_keys) |i| {
            map.insert(i, i);;
        }

        do timed(&mut results.delete_ints) {
            for uint::range(0, num_keys) |i| {
                assert map.remove(i);
            }
        }
    }
}

fn str_benchmarks<M: map::Map<~str, uint>>(make_map: fn() -> M,
                                           rng: @rand::Rng,
                                           num_keys: uint,
                                           results: &mut Results) {
    {
        let map = make_map();
        do timed(&mut results.sequential_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i, 10);
                map.insert(s, i);
            }

            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(i, 10);
                assert map.get(s) == i;
            }
        }
    }

    {
        let map = make_map();
        do timed(&mut results.random_strings) {
            for uint::range(0, num_keys) |i| {
                let s = uint::to_str(rng.next() as uint, 10);
                map.insert(s, i);
            }
        }
    }

    {
        let map = make_map();
        for uint::range(0, num_keys) |i| {
            map.insert(uint::to_str(i, 10), i);
        }
        do timed(&mut results.delete_strings) {
            for uint::range(0, num_keys) |i| {
                assert map.remove(uint::to_str(i, 10));
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

fn main(args: ~[~str]) {
    let num_keys = {
        if args.len() == 2 {
            uint::from_str(args[1]).get()
        } else {
            100 // woefully inadequate for any real measurement
        }
    };

    let seed = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    {
        let rng = rand::seeded_rng(copy seed);
        let mut results = empty_results();
        int_benchmarks::<map::HashMap<uint, uint>>(
            map::uint_hash, rng, num_keys, &mut results);
        str_benchmarks::<map::HashMap<~str, uint>>(
            map::str_hash, rng, num_keys, &mut results);
        write_results("libstd::map::hashmap", &results);
    }

    {
        let rng = rand::seeded_rng(copy seed);
        let mut results = empty_results();
        int_benchmarks::<@Mut<LinearMap<uint, uint>>>(
            || @Mut(LinearMap()),
            rng, num_keys, &mut results);
        str_benchmarks::<@Mut<LinearMap<~str, uint>>>(
            || @Mut(LinearMap()),
            rng, num_keys, &mut results);
        write_results("libstd::map::hashmap", &results);
    }
}