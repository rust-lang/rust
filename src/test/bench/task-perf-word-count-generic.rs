// xfail-test - #1038 - Can't do this safely with bare functions

/**
   A parallel word-frequency counting program.

   This is meant primarily to demonstrate Rust's MapReduce framework.

   It takes a list of files on the command line and outputs a list of
   words along with how many times each word is used.

*/

use std;

import option = option::t;
import option::some;
import option::none;
import str;
import std::treemap;
import vec;
import std::io;

import std::time;
import u64;

import task;
import task::joinable_task;
import comm;
import comm::chan;
import comm::port;
import comm::recv;
import comm::send;

fn map(&&filename: [u8], emit: map_reduce::putter<[u8], int>) {
    let f = io::file_reader(str::unsafe_from_bytes(filename));

    while true {
        alt read_word(f) {
          some(w) { emit(str::bytes(w), 1); }
          none. { break; }
        }
    }
}

fn reduce(&&_word: [u8], get: map_reduce::getter<int>) {
    let count = 0;

    while true { alt get() { some(_) { count += 1; } none. { break; } } }
}

mod map_reduce {
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter<send K, send V> = fn(K, V);

    // FIXME: the first K1 parameter should probably be a -, but that
    // doesn't parse at the moment.
    type mapper<send K1, send K2, send V> = fn(K1, putter<K2, V>);

    type getter<send V> = fn() -> option<V>;

    type reducer<send K, send V> = fn(K, getter<V>);

    tag ctrl_proto<send K, send V> {
        find_reducer(K, chan<chan<reduce_proto<V>>>);
        mapper_done;
    }

    tag reduce_proto<send V> { emit_val(V); done; ref; release; }

    fn start_mappers<send K1, send K2,
                     send V>(map: mapper<K1, K2, V>,
                         ctrl: chan<ctrl_proto<K2, V>>, inputs: [K1]) ->
       [joinable_task] {
        let tasks = [];
        for i in inputs {
            let m = map, c = ctrl, ii = i;
            tasks += [task::spawn_joinable(bind map_task(m, c, ii))];
        }
        ret tasks;
    }

    fn map_task<send K1, send K2,
                send V>(-map: mapper<K1, K2, V>,
                          -ctrl: chan<ctrl_proto<K2, V>>,
                    -input: K1) {
        // log_full(core::error, "map_task " + input);
        let intermediates = treemap::init();

        fn emit<send K2,
                send V>(im: treemap::treemap<K2, chan<reduce_proto<V>>>,
                    ctrl: chan<ctrl_proto<K2, V>>, key: K2, val: V) {
            let c;
            alt treemap::find(im, key) {
              some(_c) { c = _c; }
              none. {
                let p = port();
                send(ctrl, find_reducer(key, chan(p)));
                c = recv(p);
                treemap::insert(im, key, c);
                send(c, ref);
              }
            }
            send(c, emit_val(val));
        }

        map(input, bind emit(intermediates, ctrl, _, _));

        fn finish<send K, send V>(_k: K, v: chan<reduce_proto<V>>) {
            send(v, release);
        }
        treemap::traverse(intermediates, finish);
        send(ctrl, mapper_done);
    }

    fn reduce_task<send K,
                   send V>(-reduce: reducer<K, V>, -key: K,
                       -out: chan<chan<reduce_proto<V>>>) {
        let p = port();

        send(out, chan(p));

        let ref_count = 0;
        let is_done = false;

        fn get<send V>(p: port<reduce_proto<V>>,
                         &ref_count: int, &is_done: bool)
           -> option<V> {
            while !is_done || ref_count > 0 {
                alt recv(p) {
                  emit_val(v) {
                    // #error("received %d", v);
                    ret some(v);
                  }
                  done. {
                    // #error("all done");
                    is_done = true;
                  }
                  ref. { ref_count += 1; }
                  release. { ref_count -= 1; }
                }
            }
            ret none;
        }

        reduce(key, bind get(p, ref_count, is_done));
    }

    fn map_reduce<send K1, send K2,
                  send V>(map: mapper<K1, K2, V>, reduce: reducer<K2, V>,
                      inputs: [K1]) {
        let ctrl = port();

        // This task becomes the master control task. It task::_spawns
        // to do the rest.

        let reducers = treemap::init();

        let tasks = start_mappers(map, chan(ctrl), inputs);

        let num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            alt recv(ctrl) {
              mapper_done. {
                // #error("received mapper terminated.");
                num_mappers -= 1;
              }
              find_reducer(k, cc) {
                let c;
                // log_full(core::error, "finding reducer for " + k);
                alt treemap::find(reducers, k) {
                  some(_c) {
                    // log_full(core::error,
                    // "reusing existing reducer for " + k);
                    c = _c;
                  }
                  none. {
                    // log_full(core::error, "creating new reducer for " + k);
                    let p = port();
                    let r = reduce, kk = k;
                    tasks +=
                        [task::spawn_joinable(bind reduce_task(r, kk,
                                                               chan(p)))];
                    c = recv(p);
                    treemap::insert(reducers, k, c);
                  }
                }
                send(cc, c);
              }
            }
        }

        fn finish<send K, send V>(_k: K, v: chan<reduce_proto<V>>) {
            send(v, done);
        }
        treemap::traverse(reducers, finish);

        for t in tasks { task::join(t); }
    }
}

fn main(argv: [str]) {
    if vec::len(argv) < 2u {
        let out = io::stdout();

        out.write_line(#fmt["Usage: %s <filename> ...", argv[0]]);

        // TODO: run something just to make sure the code hasn't
        // broken yet. This is the unit test mode of this program.

        ret;
    }

    let iargs = [];
    for a in vec::slice(argv, 1u, vec::len(argv)) {
        iargs += [str::bytes(a)];
    }

    let start = time::precise_time_ns();

    map_reduce::map_reduce(map, reduce, iargs);
    let stop = time::precise_time_ns();

    let elapsed = stop - start;
    elapsed /= 1000000u64;

    log_full(core::error, "MapReduce completed in "
             + u64::str(elapsed) + "ms");
}

fn read_word(r: io::reader) -> option<str> {
    let w = "";

    while !r.eof() {
        let c = r.read_char();


        if is_word_char(c) {
            w += str::from_char(c);
        } else { if w != "" { ret some(w); } }
    }
    ret none;
}

fn is_digit(c: char) -> bool {
    alt c {
      '0' { true }
      '1' { true }
      '2' { true }
      '3' { true }
      '4' { true }
      '5' { true }
      '6' { true }
      '7' { true }
      '8' { true }
      '9' { true }
      _ { false }
    }
}

fn is_alpha_lower(c: char) -> bool {
    alt c {
      'a' { true }
      'b' { true }
      'c' { true }
      'd' { true }
      'e' { true }
      'f' { true }
      'g' { true }
      'h' { true }
      'i' { true }
      'j' { true }
      'k' { true }
      'l' { true }
      'm' { true }
      'n' { true }
      'o' { true }
      'p' { true }
      'q' { true }
      'r' { true }
      's' { true }
      't' { true }
      'u' { true }
      'v' { true }
      'w' { true }
      'x' { true }
      'y' { true }
      'z' { true }
      _ { false }
    }
}

fn is_alpha_upper(c: char) -> bool {
    alt c {
      'A' { true }
      'B' { true }
      'C' { true }
      'D' { true }
      'E' { true }
      'F' { true }
      'G' { true }
      'H' { true }
      'I' { true }
      'J' { true }
      'K' { true }
      'L' { true }
      'M' { true }
      'N' { true }
      'O' { true }
      'P' { true }
      'Q' { true }
      'R' { true }
      'S' { true }
      'T' { true }
      'U' { true }
      'V' { true }
      'W' { true }
      'X' { true }
      'Y' { true }
      'Z' { true }
      _ { false }
    }
}

fn is_alpha(c: char) -> bool { is_alpha_upper(c) || is_alpha_lower(c) }

fn is_word_char(c: char) -> bool { is_alpha(c) || is_digit(c) || c == '_' }
