// run-pass

#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(unused_mut)]
// ignore-emscripten No support for threads

/**
   A somewhat reduced test case to expose some Valgrind issues.

   This originally came from the word-count benchmark.
*/

pub fn map(filename: String, mut emit: map_reduce::putter) {
    emit(filename, "1".to_string());
}

mod map_reduce {
    use std::collections::HashMap;
    use std::sync::mpsc::{channel, Sender};
    use std::str;
    use std::thread;

    pub type putter<'a> = Box<dyn FnMut(String, String) + 'a>;

    pub type mapper = extern "C" fn(String, putter);

    enum ctrl_proto { find_reducer(Vec<u8>, Sender<isize>), mapper_done, }

    fn start_mappers(ctrl: Sender<ctrl_proto>, inputs: Vec<String>) {
        for i in &inputs {
            let ctrl = ctrl.clone();
            let i = i.clone();
            thread::spawn(move|| map_task(ctrl.clone(), i.clone()) );
        }
    }

    fn map_task(ctrl: Sender<ctrl_proto>, input: String) {
        let mut intermediates = HashMap::new();

        fn emit(im: &mut HashMap<String, isize>,
                ctrl: Sender<ctrl_proto>, key: String,
                _val: String) {
            if im.contains_key(&key) {
                return;
            }
            let (tx, rx) = channel();
            println!("sending find_reducer");
            ctrl.send(ctrl_proto::find_reducer(key.as_bytes().to_vec(), tx)).unwrap();
            println!("receiving");
            let c = rx.recv().unwrap();
            println!("{}", c);
            im.insert(key, c);
        }

        let ctrl_clone = ctrl.clone();
        ::map(input, Box::new(|a,b| emit(&mut intermediates, ctrl.clone(), a, b)));
        ctrl_clone.send(ctrl_proto::mapper_done).unwrap();
    }

    pub fn map_reduce(inputs: Vec<String>) {
        let (tx, rx) = channel();

        // This thread becomes the master control thread. It spawns others
        // to do the rest.

        let mut reducers: HashMap<String, isize>;

        reducers = HashMap::new();

        start_mappers(tx, inputs.clone());

        let mut num_mappers = inputs.len() as isize;

        while num_mappers > 0 {
            match rx.recv().unwrap() {
              ctrl_proto::mapper_done => { num_mappers -= 1; }
              ctrl_proto::find_reducer(k, cc) => {
                let mut c;
                match reducers.get(&str::from_utf8(&k).unwrap().to_string()) {
                  Some(&_c) => { c = _c; }
                  None => { c = 0; }
                }
                cc.send(c).unwrap();
              }
            }
        }
    }
}

pub fn main() {
    map_reduce::map_reduce(
        vec!["../src/test/run-pass/hashmap-memory.rs".to_string()]);
}
