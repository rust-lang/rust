// xfail-stage1
// xfail-stage2
// xfail-stage3
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

fn map(str filename, map_reduce::putter emit) {
    // log_err "mapping " + filename;
    auto f = io::file_reader(filename);

    while(true) {
        alt(read_word(f)) {
            case (some(?w)) {
                emit(w, 1);
            }
            case (none) {
                break;
            }
        }
    }
    // log_err "done mapping " + filename;
}

fn reduce(str word, map_reduce::getter get) {
    // log_err "reducing " + word;
    auto count = 0;

    while(true) {
        alt(get()) {
            some(_) {
                // log_err "received word " + word;
                count += 1;
            }
            none { break }
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

    type putter = fn(str, int) -> ();

    type mapper = fn(str, putter);

    type getter = fn() -> option[int];

    type reducer = fn(str, getter);

    tag ctrl_proto {
        find_reducer(u8[], chan[chan[reduce_proto]]);
        mapper_done;
    }

    tag reduce_proto {
        emit_val(int);
        done;
        ref;
        release;
    }

    fn start_mappers(chan[ctrl_proto] ctrl,
                     vec[str] inputs) -> task[] {
        auto tasks = ~[];
        // log_err "starting mappers";
        for(str i in inputs) {
            // log_err "starting mapper for " + i;
            tasks += ~[spawn map_task(ctrl, i)];
        }
        // log_err "done starting mappers";
        ret tasks;
    }

    fn map_task(chan[ctrl_proto] ctrl,
                str input) {
        // log_err "map_task " + input;
        auto intermediates = map::new_str_hash();

        fn emit(&map::hashmap[str, chan[reduce_proto]] im,
                chan[ctrl_proto] ctrl,
                str key, int val) {
            // log_err "emitting " + key;
            auto c;
            alt(im.find(key)) {
                some(?_c) {
                    // log_err "reusing saved channel for " + key;
                    c = _c
                }
                none {
                    // log_err "fetching new channel for " + key;
                    auto p = port[chan[reduce_proto]]();
                    auto keyi = str::bytes_ivec(key);
                    ctrl <| find_reducer(keyi, chan(p));
                    p |> c;
                    im.insert(key, clone(c));
                    c <| ref;
                }
            }
            c <| emit_val(val);
        }

        map(input, bind emit(intermediates, ctrl, _, _));

        for each(@rec(str key, chan[reduce_proto] val) kv
                 in intermediates.items()) {
            // log_err "sending done to reducer for " + kv._0;
            kv.val <| release;
        }

        ctrl <| mapper_done;

        // log_err "~map_task " + input;
    }

    fn reduce_task(str key, chan[chan[reduce_proto]] out) {
        // log_err "reduce_task " + key;
        auto p = port();

        out <| chan(p);

        auto ref_count = 0;
        auto is_done = false;

        fn get(&port[reduce_proto] p, &mutable int ref_count,
               &mutable bool is_done) -> option[int] {
            while (!is_done || ref_count > 0) {
                auto m;
                p |> m;

                alt(m) {
                    emit_val(?v) {
                        // log_err #fmt("received %d", v);
                        ret some(v);
                    }
                    done {
                        // log_err "all done";
                        is_done = true;
                    }
                    ref {
                        ref_count += 1;
                    }
                    release {
                        ref_count -= 1;
                    }
                }
            }
            ret none;
        }

        reduce(key, bind get(p, ref_count, is_done));
        // log_err "~reduce_task " + key;
    }

    fn map_reduce (vec[str] inputs) {
        auto ctrl = port[ctrl_proto]();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let map::hashmap[str, chan[reduce_proto]] reducers;

        reducers = map::new_str_hash();

        auto tasks = start_mappers(chan(ctrl), inputs);

        auto num_mappers = vec::len(inputs) as int;

        while(num_mappers > 0) {
            auto m;
            ctrl |> m;

            alt(m) {
                mapper_done {
                    // log_err "received mapper terminated.";
                    num_mappers -= 1;
                }
                find_reducer(?ki, ?cc) {
                    auto c;
                    auto k = str::unsafe_from_bytes_ivec(ki);
                    // log_err "finding reducer for " + k;
                    alt(reducers.find(k)) {
                        some(?_c) {
                            // log_err "reusing existing reducer for " + k;
                            c = _c;
                        }
                        none {
                            // log_err "creating new reducer for " + k;
                            auto p = port();
                            tasks += ~[spawn reduce_task(k, chan(p))];
                            p |> c;
                            reducers.insert(k, c);
                        }
                    }
                    cc <| clone(c);
                }
            }
        }

        for each(@rec(str key, chan[reduce_proto] val) kv
                 in reducers.items()) {
            // log_err "sending done to reducer for " + kv._0;
            kv.val <| done;
        }

        // log_err #fmt("joining %u tasks", ivec::len(tasks));
        for (task t in tasks) {
            task::join(t);
        }
        // log_err "control task done.";
    }
}

fn main(vec[str] argv) {
    if(vec::len(argv) < 2u) {
        auto out = io::stdout();

        out.write_line(#fmt("Usage: %s <filename> ...", argv.(0)));
        fail;
    }

    auto start = time::precise_time_ns();
    map_reduce::map_reduce(vec::slice(argv, 1u, vec::len(argv)));
    auto stop = time::precise_time_ns();

    auto elapsed = stop - start;
    elapsed /= 1000000u64;

    log_err "MapReduce completed in " + u64::str(elapsed) + "ms";
}

fn read_word(io::reader r) -> option[str] {
    auto w = "";

    while(!r.eof()) {
        auto c = r.read_char();

        if(is_word_char(c)) {
            w += str::from_char(c);
        }
        else {
            if(w != "") {
                ret some(w);
            }
        }
    }
    ret none;
}

fn is_digit(char c) -> bool {
    alt(c) {
        case ('0') { true }
        case ('1') { true }
        case ('2') { true }
        case ('3') { true }
        case ('4') { true }
        case ('5') { true }
        case ('6') { true }
        case ('7') { true }
        case ('8') { true }
        case ('9') { true }
        case (_) { false }
    }
}

fn is_alpha_lower (char c) -> bool {
    alt(c) {
        case ('a') { true }
        case ('b') { true }
        case ('c') { true }
        case ('d') { true }
        case ('e') { true }
        case ('f') { true }
        case ('g') { true }
        case ('h') { true }
        case ('i') { true }
        case ('j') { true }
        case ('k') { true }
        case ('l') { true }
        case ('m') { true }
        case ('n') { true }
        case ('o') { true }
        case ('p') { true }
        case ('q') { true }
        case ('r') { true }
        case ('s') { true }
        case ('t') { true }
        case ('u') { true }
        case ('v') { true }
        case ('w') { true }
        case ('x') { true }
        case ('y') { true }
        case ('z') { true }
        case (_) { false }
    }
}

fn is_alpha_upper (char c) -> bool {
    alt(c) {
        case ('A') { true }
        case ('B') { true }
        case ('C') { true }
        case ('D') { true }
        case ('E') { true }
        case ('F') { true }
        case ('G') { true }
        case ('H') { true }
        case ('I') { true }
        case ('J') { true }
        case ('K') { true }
        case ('L') { true }
        case ('M') { true }
        case ('N') { true }
        case ('O') { true }
        case ('P') { true }
        case ('Q') { true }
        case ('R') { true }
        case ('S') { true }
        case ('T') { true }
        case ('U') { true }
        case ('V') { true }
        case ('W') { true }
        case ('X') { true }
        case ('Y') { true }
        case ('Z') { true }
        case (_) { false }
    }
}

fn is_alpha(char c) -> bool {
    is_alpha_upper(c) || is_alpha_lower(c)
}

fn is_word_char(char c) -> bool {
    is_alpha(c) || is_digit(c) || c == '_'
}