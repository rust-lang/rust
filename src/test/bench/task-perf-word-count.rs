// xfail-pretty
/**
   A parallel word-frequency counting program.

   This is meant primarily to demonstrate Rust's MapReduce framework.

   It takes a list of files on the command line and outputs a list of
   words along with how many times each word is used.

*/

use std;

import std::io;
import option = std::option::t;
import std::option::some;
import std::option::none;
import std::str;
import std::vec;
import std::map;
import std::ivec;

import std::time;
import std::u64;

import std::task;
import clone = std::task::clone_chan;

fn map(filename: str, emit: map_reduce::putter) {
    // log_err "mapping " + filename;
    let f = io::file_reader(filename);


    while true {
        alt read_word(f) { some(w) { emit(w, 1); } none. { break; } }
    }
    // log_err "done mapping " + filename;
}

fn reduce(word: str, get: map_reduce::getter) {
    // log_err "reducing " + word;
    let count = 0;


    while true {
        alt get() {
          some(_) {
            // log_err "received word " + word;
            count += 1;
          }
          none. { break }
        }
    }

    // auto out = io::stdout();
    // out.write_line(#fmt("%s: %d", word, count));

    // log_err "reduce " + word + " done.";
}

mod map_reduce {
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter = fn(str, int) ;

    type mapper = fn(str, putter) ;

    type getter = fn() -> option[int] ;

    type reducer = fn(str, getter) ;

    tag ctrl_proto {
        find_reducer(u8[], chan[chan[reduce_proto]]);
        mapper_done;
    }

    tag reduce_proto { emit_val(int); done; ref; release; }

    fn start_mappers(ctrl: chan[ctrl_proto], inputs: vec[str]) -> task[] {
        let tasks = ~[];
        // log_err "starting mappers";
        for i: str  in inputs {
            // log_err "starting mapper for " + i;
            tasks += ~[spawn map_task(ctrl, i)];
        }
        // log_err "done starting mappers";
        ret tasks;
    }

    fn map_task(ctrl: chan[ctrl_proto], input: str) {
        // log_err "map_task " + input;
        let intermediates = map::new_str_hash();

        fn emit(im: &map::hashmap[str, chan[reduce_proto]],
                ctrl: chan[ctrl_proto], key: str, val: int) {
            // log_err "emitting " + key;
            let c;
            alt im.find(key) {
              some(_c) {

                // log_err "reusing saved channel for " + key;
                c = _c
              }
              none. {
                // log_err "fetching new channel for " + key;
                let p = port[chan[reduce_proto]]();
                let keyi = str::bytes_ivec(key);
                ctrl <| find_reducer(keyi, chan(p));
                p |> c;
                im.insert(key, clone(c));
                c <| ref;
              }
            }
            c <| emit_val(val);
        }

        map(input, bind emit(intermediates, ctrl, _, _));

        for each kv: @{key: str, val: chan[reduce_proto]}  in
                 intermediates.items() {
            // log_err "sending done to reducer for " + kv._0;
            kv.val <| release;
        }

        ctrl <| mapper_done;

        // log_err "~map_task " + input;
    }

    fn reduce_task(key: str, out: chan[chan[reduce_proto]]) {
        // log_err "reduce_task " + key;
        let p = port();

        out <| chan(p);

        let ref_count = 0;
        let is_done = false;

        fn get(p: &port[reduce_proto], ref_count: &mutable int,
               is_done: &mutable bool) -> option[int] {
            while !is_done || ref_count > 0 {
                let m;
                p |> m;


                alt m {
                  emit_val(v) {
                    // log_err #fmt("received %d", v);
                    ret some(v);
                  }
                  done. {
                    // log_err "all done";
                    is_done = true;
                  }
                  ref. { ref_count += 1; }
                  release. { ref_count -= 1; }
                }
            }
            ret none;
        }

        reduce(key, bind get(p, ref_count, is_done));
        // log_err "~reduce_task " + key;
    }

    fn map_reduce(inputs: vec[str]) {
        let ctrl = port[ctrl_proto]();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let reducers: map::hashmap[str, chan[reduce_proto]];

        reducers = map::new_str_hash();

        let tasks = start_mappers(chan(ctrl), inputs);

        let num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            let m;
            ctrl |> m;

            alt m {
              mapper_done. {
                // log_err "received mapper terminated.";
                num_mappers -= 1;
              }
              find_reducer(ki, cc) {
                let c;
                let k = str::unsafe_from_bytes_ivec(ki);
                // log_err "finding reducer for " + k;
                alt reducers.find(k) {
                  some(_c) {
                    // log_err "reusing existing reducer for " + k;
                    c = _c;
                  }
                  none. {
                    // log_err "creating new reducer for " + k;
                    let p = port();
                    tasks += ~[spawn reduce_task(k, chan(p))];
                    p |> c;
                    reducers.insert(k, c);
                  }
                }
                cc <| clone(c);
              }
            }
        }

        for each kv: @{key: str, val: chan[reduce_proto]}  in reducers.items()
                 {
            // log_err "sending done to reducer for " + kv._0;
            kv.val <| done;
        }


        // log_err #fmt("joining %u tasks", ivec::len(tasks));
        for t: task  in tasks { task::join(t); }
        // log_err "control task done.";
    }
}

fn main(argv: vec[str]) {
    if vec::len(argv) < 2u {
        let out = io::stdout();

        out.write_line(#fmt("Usage: %s <filename> ...", argv.(0)));

        // TODO: run something just to make sure the code hasn't
        // broken yet. This is the unit test mode of this program.

        ret;
    }

    // We can get by with 8k stacks, and we'll probably exhaust our
    // address space otherwise.
    task::set_min_stack(8192u);

    let start = time::precise_time_ns();

    map_reduce::map_reduce(vec::slice(argv, 1u, vec::len(argv)));
    let stop = time::precise_time_ns();

    let elapsed = stop - start;
    elapsed /= 1000000u64;

    log_err "MapReduce completed in " + u64::str(elapsed) + "ms";
}

fn read_word(r: io::reader) -> option[str] {
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
