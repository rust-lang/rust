/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

use std;

import option = option;
import option::some;
import option::none;
import str;
import vec;
import std::map;
import std::map::hashmap;
import task;
import comm::Chan;
import comm::chan;
import comm::port;
import comm::send;
import comm::recv;
import comm;

fn map(filename: ~str, emit: map_reduce::putter) { emit(filename, ~"1"); }

mod map_reduce {
    export putter;
    export mapper;
    export map_reduce;

    type putter = fn@(~str, ~str);

    type mapper = extern fn(~str, putter);

    enum ctrl_proto { find_reducer(~[u8], Chan<int>), mapper_done, }

    fn start_mappers(ctrl: Chan<ctrl_proto>, inputs: ~[~str]) {
        for inputs.each |i| {
            task::spawn(|| map_task(ctrl, i) );
        }
    }

    fn map_task(ctrl: Chan<ctrl_proto>, input: ~str) {
        let intermediates = map::str_hash();

        fn emit(im: map::hashmap<~str, int>, ctrl: Chan<ctrl_proto>, key: ~str,
                val: ~str) {
            let mut c;
            match im.find(key) {
              some(_c) => { c = _c }
              none => {
                let p = port();
                error!("sending find_reducer");
                send(ctrl, find_reducer(str::to_bytes(key), chan(p)));
                error!("receiving");
                c = recv(p);
                log(error, c);
                im.insert(key, c);
              }
            }
        }

        map(input, |a,b| emit(intermediates, ctrl, a, b) );
        send(ctrl, mapper_done);
    }

    fn map_reduce(inputs: ~[~str]) {
        let ctrl = port();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let mut reducers: map::hashmap<~str, int>;

        reducers = map::str_hash();

        start_mappers(chan(ctrl), inputs);

        let mut num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            match recv(ctrl) {
              mapper_done => { num_mappers -= 1; }
              find_reducer(k, cc) => {
                let mut c;
                match reducers.find(str::from_bytes(k)) {
                  some(_c) => { c = _c; }
                  none => { c = 0; }
                }
                send(cc, c);
              }
            }
        }
    }
}

fn main() {
    map_reduce::map_reduce(~[~"../src/test/run-pass/hashmap-memory.rs"]);
}
