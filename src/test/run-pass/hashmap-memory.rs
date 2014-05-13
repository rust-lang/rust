
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

extern crate collections;


/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

pub fn map(filename: StrBuf, emit: map_reduce::putter) {
    emit(filename, "1".to_strbuf());
}

mod map_reduce {
    use collections::HashMap;
    use std::str;
    use std::task;

    pub type putter<'a> = |StrBuf, StrBuf|: 'a;

    pub type mapper = extern fn(StrBuf, putter);

    enum ctrl_proto { find_reducer(Vec<u8>, Sender<int>), mapper_done, }

    fn start_mappers(ctrl: Sender<ctrl_proto>, inputs: Vec<StrBuf>) {
        for i in inputs.iter() {
            let ctrl = ctrl.clone();
            let i = i.clone();
            task::spawn(proc() map_task(ctrl.clone(), i.clone()) );
        }
    }

    fn map_task(ctrl: Sender<ctrl_proto>, input: StrBuf) {
        let mut intermediates = HashMap::new();

        fn emit(im: &mut HashMap<StrBuf, int>,
                ctrl: Sender<ctrl_proto>, key: StrBuf,
                _val: StrBuf) {
            if im.contains_key(&key) {
                return;
            }
            let (tx, rx) = channel();
            println!("sending find_reducer");
            ctrl.send(find_reducer(Vec::from_slice(key.as_bytes()), tx));
            println!("receiving");
            let c = rx.recv();
            println!("{:?}", c);
            im.insert(key, c);
        }

        let ctrl_clone = ctrl.clone();
        ::map(input, |a,b| emit(&mut intermediates, ctrl.clone(), a, b) );
        ctrl_clone.send(mapper_done);
    }

    pub fn map_reduce(inputs: Vec<StrBuf>) {
        let (tx, rx) = channel();

        // This task becomes the master control task. It spawns others
        // to do the rest.

        let mut reducers: HashMap<StrBuf, int>;

        reducers = HashMap::new();

        start_mappers(tx, inputs.clone());

        let mut num_mappers = inputs.len() as int;

        while num_mappers > 0 {
            match rx.recv() {
              mapper_done => { num_mappers -= 1; }
              find_reducer(k, cc) => {
                let mut c;
                match reducers.find(&str::from_utf8(
                        k.as_slice()).unwrap().to_strbuf()) {
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
    map_reduce::map_reduce(
        vec!("../src/test/run-pass/hashmap-memory.rs".to_strbuf()));
}
