// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

extern mod std;

use std::oldmap;
use std::oldmap::HashMap;
use core::pipes::*;

pub fn map(filename: ~str, emit: map_reduce::putter) { emit(filename, ~"1"); }

mod map_reduce {
    use std::oldmap;
    use std::oldmap::HashMap;
    use core::pipes::*;

    pub type putter = fn@(~str, ~str);

    pub type mapper = extern fn(~str, putter);

    enum ctrl_proto { find_reducer(~[u8], Chan<int>), mapper_done, }

    fn start_mappers(ctrl: SharedChan<ctrl_proto>, inputs: ~[~str]) {
        for inputs.each |i| {
            let ctrl = ctrl.clone();
            let i = copy *i;
            task::spawn(|move i| map_task(ctrl.clone(), copy i) );
        }
    }

    fn map_task(ctrl: SharedChan<ctrl_proto>, input: ~str) {
        let intermediates = oldmap::HashMap();

        fn emit(im: oldmap::HashMap<~str, int>, ctrl: SharedChan<ctrl_proto>, key: ~str,
                val: ~str) {
            let mut c;
            match im.find(copy key) {
              Some(_c) => { c = _c }
              None => {
                  let (pp, cc) = stream();
                error!("sending find_reducer");
                ctrl.send(find_reducer(str::to_bytes(key), cc));
                error!("receiving");
                c = pp.recv();
                log(error, c);
                im.insert(key, c);
              }
            }
        }

        let ctrl_clone = ctrl.clone();
        ::map(input, |a,b| emit(intermediates, ctrl.clone(), a, b) );
        ctrl_clone.send(mapper_done);
    }

    pub fn map_reduce(inputs: ~[~str]) {
        let (ctrl_port, ctrl_chan) = stream();
        let ctrl_chan = SharedChan(ctrl_chan);

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let mut reducers: oldmap::HashMap<~str, int>;

        reducers = oldmap::HashMap();

        start_mappers(ctrl_chan, copy inputs);

        let mut num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            match ctrl_port.recv() {
              mapper_done => { num_mappers -= 1; }
              find_reducer(k, cc) => {
                let mut c;
                match reducers.find(str::from_bytes(k)) {
                  Some(_c) => { c = _c; }
                  None => { c = 0; }
                }
                cc.send(c);
              }
            }
        }
    }
}

pub fn main() {
    map_reduce::map_reduce(~[~"../src/test/run-pass/hashmap-memory.rs"]);
}
