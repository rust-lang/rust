// xfail-pretty
/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

use std;

import std::io;
import option = std::option::t;
import std::option::some;
import std::option::none;
import std::str;
import std::vec;
import std::map;

fn map(filename: str, emit: map_reduce::putter) { emit(filename, "1"); }

mod map_reduce {
    export putter;
    export mapper;
    export map_reduce;

    type putter = fn(str, str) ;

    type mapper = fn(str, putter) ;

    tag ctrl_proto { find_reducer(str, chan[int]); mapper_done; }

    fn start_mappers(ctrl: chan[ctrl_proto], inputs: vec[str]) {
        for i: str  in inputs { spawn map_task(ctrl, i); }
    }

    fn map_task(ctrl: chan[ctrl_proto], input: str) {

        let intermediates = map::new_str_hash();

        fn emit(im: &map::hashmap[str, int], ctrl: chan[ctrl_proto], key: str,
                val: str) {
            let c;
            alt im.find(key) {
              some(_c) { c = _c }
              none. {
                let p = port();
                log_err "sending find_reducer";
                ctrl <| find_reducer(key, chan(p));
                log_err "receiving";
                p |> c;
                log_err c;
                im.insert(key, c);
              }
            }
        }

        map(input, bind emit(intermediates, ctrl, _, _));
        ctrl <| mapper_done;
    }

    fn map_reduce(inputs: vec[str]) {
        let ctrl = port[ctrl_proto]();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let reducers: map::hashmap[str, int];

        reducers = map::new_str_hash();

        start_mappers(chan(ctrl), inputs);

        let num_mappers = vec::len(inputs) as int;


        while num_mappers > 0 {
            let m;
            ctrl |> m;


            alt m {
              mapper_done. { num_mappers -= 1; }
              find_reducer(k, cc) {
                let c;
                alt reducers.find(k) { some(_c) { c = _c; } none. { c = 0; } }
                cc <| c;
              }
            }
        }
    }
}

fn main() {
    map_reduce::map_reduce(["../src/test/run-pass/hashmap-memory.rs"]);
}