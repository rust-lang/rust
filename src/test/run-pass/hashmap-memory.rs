// ignore-fast

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

pub fn map(filename: ~str, emit: map_reduce::putter) { emit(filename, ~"1"); }

mod map_reduce {
    use std::hashmap::HashMap;
    use std::str;
    use std::task;

    pub type putter<'a> = 'a |~str, ~str|;

    pub type mapper = extern fn(~str, putter);

    enum ctrl_proto { find_reducer(~[u8], Chan<int>), mapper_done, }

    fn start_mappers(ctrl: Chan<ctrl_proto>, inputs: ~[~str]) {
        for i in inputs.iter() {
            let ctrl = ctrl.clone();
            let i = i.clone();
            task::spawn(proc() map_task(ctrl.clone(), i.clone()) );
        }
    }

    fn map_task(ctrl: Chan<ctrl_proto>, input: ~str) {
        let mut intermediates = HashMap::new();

        fn emit(im: &mut HashMap<~str, int>,
                ctrl: Chan<ctrl_proto>, key: ~str,
                _val: ~str) {
            if im.contains_key(&key) {
                return;
            }
            let (pp, cc) = Chan::new();
            error!("sending find_reducer");
            ctrl.send(find_reducer(key.as_bytes().to_owned(), cc));
            error!("receiving");
            let c = pp.recv();
            error!("{:?}", c);
            im.insert(key, c);
        }

        let ctrl_clone = ctrl.clone();
        ::map(input, |a,b| emit(&mut intermediates, ctrl.clone(), a, b) );
        ctrl_clone.send(mapper_done);
    }

    pub fn map_reduce(inputs: ~[~str]) {
        let (ctrl_port, ctrl_chan) = Chan::new();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let mut reducers: HashMap<~str, int>;

        reducers = HashMap::new();

        start_mappers(ctrl_chan, inputs.clone());

        let mut num_mappers = inputs.len() as int;

        while num_mappers > 0 {
            match ctrl_port.recv() {
              mapper_done => { num_mappers -= 1; }
              find_reducer(k, cc) => {
                let mut c;
                match reducers.find(&str::from_utf8(k).unwrap().to_owned()) {
                  Some(&_c) => { c = _c; }
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
