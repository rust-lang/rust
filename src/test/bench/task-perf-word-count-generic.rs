/**
   A parallel word-frequency counting program.

   This is meant primarily to demonstrate Rust's MapReduce framework.

   It takes a list of files on the command line and outputs a list of
   words along with how many times each word is used.

*/

// xfail-pretty

use std;

import option = option;
import option::some;
import option::none;
import str;
import std::map;
import std::map::hashmap;
import vec;
import io;
import io::WriterUtil;

import std::time;
import u64;

import task;
import comm;
import comm::Chan;
import comm::chan;
import comm::Port;
import comm::port;
import comm::recv;
import comm::send;

macro_rules! move_out {
    { $x:expr } => { unsafe { let y <- *ptr::addr_of($x); y } }
}

trait word_reader {
    fn read_word() -> option<~str>;
}

trait hash_key {
    pure fn hash() -> uint;
    pure fn eq(&&k: self) -> bool;
}

fn mk_hash<K: const hash_key, V: copy>() -> map::hashmap<K, V> {
    pure fn hashfn<K: const hash_key>(k: &K) -> uint { k.hash() }
    pure fn hasheq<K: const hash_key>(k1: &K, k2: &K) -> bool { k1.eq(*k2) }

    map::hashmap(hashfn, hasheq)
}

impl ~str: hash_key {
    pure fn hash() -> uint { str::hash(&self) }
    pure fn eq(&&x: ~str) -> bool { self == x }
}

// These used to be in task, but they disappeard.
type joinable_task = Port<()>;
fn spawn_joinable(+f: fn~()) -> joinable_task {
    let p = port();
    let c = chan(p);
    do task::spawn() |move f| {
        f();
        c.send(());
    }
    p
}

fn join(t: joinable_task) {
    t.recv()
}

impl io::Reader: word_reader {
    fn read_word() -> option<~str> { read_word(self) }
}

fn file_word_reader(filename: ~str) -> word_reader {
    match io::file_reader(filename) {
      result::ok(f) => { f as word_reader }
      result::err(e) => { fail fmt!{"%?", e} }
    }
}

fn map(f: fn~() -> word_reader, emit: map_reduce::putter<~str, int>) {
    let f = f();
    loop {
        match f.read_word() {
          some(w) => { emit(w, 1); }
          none => { break; }
        }
    }
}

fn reduce(&&word: ~str, get: map_reduce::getter<int>) {
    let mut count = 0;

    loop { match get() { some(_) => { count += 1; } none => { break; } } }
    
    io::println(fmt!{"%s\t%?", word, count});
}

struct box<T> {
    let mut contents: option<T>;
    new(+x: T) { self.contents = some(x); }

    fn swap(f: fn(+T) -> T) {
        let mut tmp = none;
        self.contents <-> tmp;
        self.contents = some(f(option::unwrap(tmp)));
    }

    fn unwrap() -> T {
        let mut tmp = none;
        self.contents <-> tmp;
        option::unwrap(tmp)
    }
}

mod map_reduce {
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter<K: send, V: send> = fn(K, V);

    type mapper<K1: send, K2: send, V: send> = fn~(K1, putter<K2, V>);

    type getter<V: send> = fn() -> option<V>;

    type reducer<K: copy send, V: copy send> = fn~(K, getter<V>);

    enum ctrl_proto<K: copy send, V: copy send> {
        find_reducer(K, Chan<Chan<reduce_proto<V>>>),
        mapper_done
    }


    proto! ctrl_proto {
        open: send<K: copy send, V: copy send> {
            find_reducer(K) -> reducer_response<K, V>,
            mapper_done -> !
        }

        reducer_response: recv<K: copy send, V: copy send> {
            reducer(Chan<reduce_proto<V>>) -> open<K, V>
        }
    }

    enum reduce_proto<V: copy send> { emit_val(V), done, addref, release }

    fn start_mappers<K1: copy send, K2: const copy send hash_key,
                     V: copy send>(
        map: mapper<K1, K2, V>,
        &ctrls: ~[ctrl_proto::server::open<K2, V>],
        inputs: ~[K1])
        -> ~[joinable_task]
    {
        let mut tasks = ~[];
        for inputs.each |i| {
            let (ctrl, ctrl_server) = ctrl_proto::init();
            let ctrl = box(ctrl);
            vec::push(tasks, spawn_joinable(|| map_task(map, ctrl, i) ));
            vec::push(ctrls, ctrl_server);
        }
        return tasks;
    }

    fn map_task<K1: copy send, K2: const copy send hash_key, V: copy send>(
        map: mapper<K1, K2, V>,
        ctrl: box<ctrl_proto::client::open<K2, V>>,
        input: K1)
    {
        // log(error, "map_task " + input);
        let intermediates = mk_hash();

        do map(input) |key, val| {
            let mut c = none;
            match intermediates.find(key) {
              some(_c) => { c = some(_c); }
              none => {
                do ctrl.swap |ctrl| {
                    let ctrl = ctrl_proto::client::find_reducer(ctrl, key);
                    match pipes::recv(ctrl) {
                      ctrl_proto::reducer(c_, ctrl) => {
                        c = some(c_);
                        move_out!{ctrl}
                      }
                    }
                }
                intermediates.insert(key, c.get());
                send(c.get(), addref);
              }
            }
            send(c.get(), emit_val(val));
        }

        fn finish<K: copy send, V: copy send>(_k: K, v: Chan<reduce_proto<V>>)
        {
            send(v, release);
        }
        for intermediates.each_value |v| { send(v, release) }
        ctrl_proto::client::mapper_done(ctrl.unwrap());
    }

    fn reduce_task<K: copy send, V: copy send>(
        reduce: reducer<K, V>, 
        key: K,
        out: Chan<Chan<reduce_proto<V>>>)
    {
        let p = port();

        send(out, chan(p));

        let mut ref_count = 0;
        let mut is_done = false;

        fn get<V: copy send>(p: Port<reduce_proto<V>>,
                             &ref_count: int, &is_done: bool)
           -> option<V> {
            while !is_done || ref_count > 0 {
                match recv(p) {
                  emit_val(v) => {
                    // error!{"received %d", v};
                    return some(v);
                  }
                  done => {
                    // error!{"all done"};
                    is_done = true;
                  }
                  addref => { ref_count += 1; }
                  release => { ref_count -= 1; }
                }
            }
            return none;
        }

        reduce(key, || get(p, ref_count, is_done) );
    }

    fn map_reduce<K1: copy send, K2: const copy send hash_key, V: copy send>(
        map: mapper<K1, K2, V>,
        reduce: reducer<K2, V>,
        inputs: ~[K1])
    {
        let mut ctrl = ~[];

        // This task becomes the master control task. It task::_spawns
        // to do the rest.

        let reducers = mk_hash();
        let mut tasks = start_mappers(map, ctrl, inputs);
        let mut num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            let (_ready, message, ctrls) = pipes::select(ctrl);
            match option::unwrap(message) {
              ctrl_proto::mapper_done => {
                // error!{"received mapper terminated."};
                num_mappers -= 1;
                ctrl = ctrls;
              }
              ctrl_proto::find_reducer(k, cc) => {
                let c;
                // log(error, "finding reducer for " + k);
                match reducers.find(k) {
                  some(_c) => {
                    // log(error,
                    // "reusing existing reducer for " + k);
                    c = _c;
                  }
                  none => {
                    // log(error, "creating new reducer for " + k);
                    let p = port();
                    let ch = chan(p);
                    let r = reduce, kk = k;
                    vec::push(tasks,
                              spawn_joinable(|| reduce_task(r, kk, ch) ));
                    c = recv(p);
                    reducers.insert(k, c);
                  }
                }
                ctrl = vec::append_one(
                    ctrls,
                    ctrl_proto::server::reducer(move_out!{cc}, c));
              }
            }
        }

        for reducers.each_value |v| { send(v, done) }

        for tasks.each |t| { join(t); }
    }
}

fn main(argv: ~[~str]) {
    if vec::len(argv) < 2u && !os::getenv(~"RUST_BENCH").is_some() {
        let out = io::stdout();

        out.write_line(fmt!{"Usage: %s <filename> ...", argv[0]});

        return;
    }

    let readers: ~[fn~() -> word_reader]  = if argv.len() >= 2 {
        vec::view(argv, 1u, argv.len()).map(
            |f| fn~() -> word_reader { file_word_reader(f) } )
    }
    else {
        let num_readers = 50;
        let words_per_reader = 600;
        vec::from_fn(
            num_readers,
            |_i| fn~() -> word_reader {
                random_word_reader(words_per_reader) as word_reader
            })
    };

    let start = time::precise_time_ns();

    map_reduce::map_reduce(map, reduce, readers);
    let stop = time::precise_time_ns();

    let elapsed = (stop - start) / 1000000u64;

    log(error, ~"MapReduce completed in "
             + u64::str(elapsed) + ~"ms");
}

fn read_word(r: io::Reader) -> option<~str> {
    let mut w = ~"";

    while !r.eof() {
        let c = r.read_char();

        if is_word_char(c) {
            w += str::from_char(c);
        } else { if w != ~"" { return some(w); } }
    }
    return none;
}

fn is_word_char(c: char) -> bool {
    char::is_alphabetic(c) || char::is_digit(c) || c == '_'
}

struct random_word_reader: word_reader {
    let mut remaining: uint;
    let rng: rand::Rng;
    new(count: uint) {
        self.remaining = count;
        self.rng = rand::rng();
    }

    fn read_word() -> option<~str> {
        if self.remaining > 0 {
            self.remaining -= 1;
            let len = self.rng.gen_uint_range(1, 4);
            some(self.rng.gen_str(len))
        }
        else { none }
    }
}
