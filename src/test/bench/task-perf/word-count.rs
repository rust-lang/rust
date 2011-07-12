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

fn map(str filename, map_reduce::putter emit) {
    auto f = io::file_reader(filename);

    while(true) {
        alt(read_word(f)) {
            case (some(?w)) {
                emit(w, "1");
            }
            case (none) {
                break;
            }
        }
    }
}

fn reduce(str word, map_reduce::getter get) {
    auto count = 0;

    while(true) {
        alt(get()) {
            case(some(_)) { count += 1 }
            case(none) { break }
        }
    }

    auto out = io::stdout();
    out.write_line(#fmt("%s: %d", word, count));
}

mod map_reduce {
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter = fn(str, str) -> ();

    type mapper = fn(str, putter);

    type getter = fn() -> option[str];

    type reducer = fn(str, getter);

    tag ctrl_proto {
        find_reducer(str, chan[chan[reduce_proto]]);
        mapper_done;
    }

    tag reduce_proto {
        emit_val(str);
        done;
    }

    fn start_mappers(chan[ctrl_proto] ctrl,
                     vec[str] inputs) {
        for(str i in inputs) {
            spawn map_task(ctrl, i);
        }
    }

    fn map_task(chan[ctrl_proto] ctrl,
                str input) {

        auto intermediates = map::new_str_hash();

        fn emit(&map::hashmap[str, chan[reduce_proto]] im,
                chan[ctrl_proto] ctrl,
                str key, str val) {
            auto c;
            alt(im.find(key)) {
                case(some(?_c)) {
                    c = _c
                }
                case(none) {
                    auto p = port[chan[reduce_proto]]();
                    ctrl <| find_reducer(key, chan(p));
                    p |> c;
                    im.insert(key, c);
                }
            }
            c <| emit_val(val);
        }

        map(input, bind emit(intermediates, ctrl, _, _));
        ctrl <| mapper_done;
    }

    fn reduce_task(str key, chan[chan[reduce_proto]] out) {
        auto p = port();

        out <| chan(p);

        fn get(port[reduce_proto] p) -> option[str] {
            auto m;
            p |> m;

            alt(m) {
                case(emit_val(?v)) { ret some(v); }
                case(done) { ret none; }
            }
        }

        reduce(key, bind get(p));
    }

    fn map_reduce (vec[str] inputs) {
        auto ctrl = port[ctrl_proto]();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let map::hashmap[str, chan[reduce_proto]] reducers;

        reducers = map::new_str_hash();

        start_mappers(chan(ctrl), inputs);

        auto num_mappers = vec::len(inputs) as int;

        while(num_mappers > 0) {
            auto m;
            ctrl |> m;

            alt(m) {
                case(mapper_done) { num_mappers -= 1; }
                case(find_reducer(?k, ?cc)) {
                    auto c;
                    alt(reducers.find(k)) {
                        case(some(?_c)) { c = _c; }
                        case(none) {
                            auto p = port();
                            spawn reduce_task(k, chan(p));
                            p |> c;
                            reducers.insert(k, c);
                        }
                    }
                    cc <| c;
                }
            }
        }

        for each(@tup(str, chan[reduce_proto]) kv in reducers.items()) {
            kv._1 <| done;
        }
    }
}

fn main(vec[str] argv) {
    if(vec::len(argv) < 2u) {
        auto out = io::stdout();

        out.write_line(#fmt("Usage: %s <filename> ...", argv.(0)));
        fail;
    }

    map_reduce::map_reduce(vec::slice(argv, 1u, vec::len(argv)));
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