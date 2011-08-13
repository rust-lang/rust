/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

use std;

import option = std::option::t;
import std::option::some;
import std::option::none;
import std::str;
import std::ivec;
import std::map;
import std::task;
import std::comm::_chan;
import std::comm::_port;
import std::comm::send;
import std::comm::mk_port;
import std::comm;

fn map(filename: str, emit: map_reduce::putter) { emit(filename, "1"); }

mod map_reduce {
    export putter;
    export mapper;
    export map_reduce;

    type putter = fn(str, str) ;

    type mapper = fn(str, putter) ;

    tag ctrl_proto { find_reducer([u8], _chan[int]); mapper_done; }

    fn start_mappers(ctrl: _chan[ctrl_proto], inputs: &[str]) {
        for i: str  in inputs { task::_spawn(bind map_task(ctrl, i)); }
    }

    fn map_task(ctrl: _chan[ctrl_proto], input: str) {

        let intermediates = map::new_str_hash();

        fn emit(im: &map::hashmap[str, int], ctrl: _chan[ctrl_proto],
                key: str, val: str) {
            let c;
            alt im.find(key) {
              some(_c) { c = _c }
              none. {
                let p = mk_port();
                log_err "sending find_reducer";
                send(ctrl, find_reducer(str::bytes(key), p.mk_chan()));
                log_err "receiving";
                c = p.recv();
                log_err c;
                im.insert(key, c);
              }
            }
        }

        map(input, bind emit(intermediates, ctrl, _, _));
        send(ctrl, mapper_done);
    }

    fn map_reduce(inputs: &[str]) {
        let ctrl = mk_port[ctrl_proto]();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let reducers: map::hashmap[str, int];

        reducers = map::new_str_hash();

        start_mappers(ctrl.mk_chan(), inputs);

        let num_mappers = ivec::len(inputs) as int;

        while num_mappers > 0 {
            alt ctrl.recv() {
              mapper_done. { num_mappers -= 1; }
              find_reducer(k, cc) {
                let c;
                alt reducers.find(str::unsafe_from_bytes(k)) {
                  some(_c) { c = _c; }
                  none. { c = 0; }
                }
                send(cc, c);
              }
            }
        }
    }
}

fn main() {
    map_reduce::map_reduce(~["../src/test/run-pass/hashmap-memory.rs"]);
}