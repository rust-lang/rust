/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

extern mod std;

use option::Some;
use option::None;
use std::map;
use std::map::HashMap;
use comm::Chan;
use comm::Port;
use comm::send;
use comm::recv;

fn map(filename: ~str, emit: map_reduce::putter) { emit(filename, ~"1"); }

mod map_reduce {
    #[legacy_exports];
    export putter;
    export mapper;
    export map_reduce;

    type putter = fn@(~str, ~str);

    type mapper = extern fn(~str, putter);

    enum ctrl_proto { find_reducer(~[u8], Chan<int>), mapper_done, }

    fn start_mappers(ctrl: Chan<ctrl_proto>, inputs: ~[~str]) {
        for inputs.each |i| {
            let i = *i;
            task::spawn(|move i| map_task(ctrl, i) );
        }
    }

    fn map_task(ctrl: Chan<ctrl_proto>, input: ~str) {
        let intermediates = map::HashMap();

        fn emit(im: map::HashMap<~str, int>, ctrl: Chan<ctrl_proto>, key: ~str,
                val: ~str) {
            let mut c;
            match im.find(copy key) {
              Some(_c) => { c = _c }
              None => {
                let p = Port();
                error!("sending find_reducer");
                send(ctrl, find_reducer(str::to_bytes(key), Chan(&p)));
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
        let ctrl = Port();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let mut reducers: map::HashMap<~str, int>;

        reducers = map::HashMap();

        start_mappers(Chan(&ctrl), copy inputs);

        let mut num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            match recv(ctrl) {
              mapper_done => { num_mappers -= 1; }
              find_reducer(k, cc) => {
                let mut c;
                match reducers.find(str::from_bytes(k)) {
                  Some(_c) => { c = _c; }
                  None => { c = 0; }
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
