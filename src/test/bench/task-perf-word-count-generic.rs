/*!
   A parallel word-frequency counting program.

   This is meant primarily to demonstrate Rust's MapReduce framework.

   It takes a list of files on the command line and outputs a list of
   words along with how many times each word is used.

*/

// xfail-pretty

extern mod std;

use option = option;
use option::Some;
use option::None;
use std::map;
use std::map::HashMap;
use hash::Hash;
use io::{ReaderUtil, WriterUtil};

use std::time;

use comm::Chan;
use comm::Port;
use comm::recv;
use comm::send;
use cmp::Eq;
use to_bytes::IterBytes;

macro_rules! move_out (
    { $x:expr } => { unsafe { let y <- *ptr::addr_of(&($x)); move y } }
)

trait word_reader {
    fn read_word() -> Option<~str>;
}

// These used to be in task, but they disappeard.
type joinable_task = Port<()>;
fn spawn_joinable(+f: fn~()) -> joinable_task {
    let p = Port();
    let c = Chan(&p);
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
    fn read_word() -> Option<~str> { read_word(self) }
}

fn file_word_reader(filename: ~str) -> word_reader {
    match io::file_reader(&Path(filename)) {
      result::Ok(f) => { f as word_reader }
      result::Err(e) => { fail fmt!("%?", e) }
    }
}

fn map(f: fn~() -> word_reader, emit: map_reduce::putter<~str, int>) {
    let f = f();
    loop {
        match f.read_word() {
          Some(w) => { emit(&w, 1); }
          None => { break; }
        }
    }
}

fn reduce(word: &~str, get: map_reduce::getter<int>) {
    let mut count = 0;

    loop { match get() { Some(_) => { count += 1; } None => { break; } } }

    io::println(fmt!("%s\t%?", *word, count));
}

struct box<T> {
    mut contents: Option<T>,
}

impl<T> box<T> {
    fn swap(f: fn(+v: T) -> T) {
        let mut tmp = None;
        self.contents <-> tmp;
        self.contents = Some(f(option::unwrap(move tmp)));
    }

    fn unwrap() -> T {
        let mut tmp = None;
        self.contents <-> tmp;
        option::unwrap(move tmp)
    }
}

fn box<T>(+x: T) -> box<T> {
    box {
        contents: Some(move x)
    }
}

mod map_reduce {
    #[legacy_exports];
    export putter;
    export getter;
    export mapper;
    export reducer;
    export map_reduce;

    type putter<K: Send, V: Send> = fn(&K, V);

    type mapper<K1: Send, K2: Send, V: Send> = fn~(K1, putter<K2, V>);

    type getter<V: Send> = fn() -> Option<V>;

    type reducer<K: Copy Send, V: Copy Send> = fn~(&K, getter<V>);

    enum ctrl_proto<K: Copy Send, V: Copy Send> {
        find_reducer(K, Chan<Chan<reduce_proto<V>>>),
        mapper_done
    }


    proto! ctrl_proto (
        open: send<K: Copy Send, V: Copy Send> {
            find_reducer(K) -> reducer_response<K, V>,
            mapper_done -> !
        }

        reducer_response: recv<K: Copy Send, V: Copy Send> {
            reducer(Chan<reduce_proto<V>>) -> open<K, V>
        }
    )

    enum reduce_proto<V: Copy Send> { emit_val(V), done, addref, release }

    fn start_mappers<K1: Copy Send, K2: Hash IterBytes Eq Const Copy Send,
                     V: Copy Send>(
        map: &mapper<K1, K2, V>,
        ctrls: &mut ~[ctrl_proto::server::open<K2, V>],
        inputs: &~[K1])
        -> ~[joinable_task]
    {
        let mut tasks = ~[];
        for inputs.each |i| {
            let (ctrl, ctrl_server) = ctrl_proto::init();
            let ctrl = box(move ctrl);
            let i = copy *i;
            let m = copy *map;
            tasks.push(spawn_joinable(|move ctrl, move i| map_task(m, &ctrl, i)));
            ctrls.push(move ctrl_server);
        }
        move tasks
    }

    fn map_task<K1: Copy Send, K2: Hash IterBytes Eq Const Copy Send, V: Copy Send>(
        map: mapper<K1, K2, V>,
        ctrl: &box<ctrl_proto::client::open<K2, V>>,
        input: K1)
    {
        // log(error, "map_task " + input);
        let intermediates: HashMap<K2, Chan<reduce_proto<V>>>
            = map::HashMap();

        do map(input) |key: &K2, val| {
            let mut c = None;
            let found: Option<Chan<reduce_proto<V>>>
                = intermediates.find(*key);
            match found {
              Some(_c) => { c = Some(_c); }
              None => {
                do ctrl.swap |ctrl| {
                    let ctrl = ctrl_proto::client::find_reducer(move ctrl, *key);
                    match pipes::recv(move ctrl) {
                      ctrl_proto::reducer(c_, ctrl) => {
                        c = Some(c_);
                        move_out!(ctrl)
                      }
                    }
                }
                intermediates.insert(*key, c.get());
                send(c.get(), addref);
              }
            }
            send(c.get(), emit_val(val));
        }

        fn finish<K: Copy Send, V: Copy Send>(_k: K, v: Chan<reduce_proto<V>>)
        {
            send(v, release);
        }
        for intermediates.each_value |v| { send(v, release) }
        ctrl_proto::client::mapper_done(ctrl.unwrap());
    }

    fn reduce_task<K: Copy Send, V: Copy Send>(
        reduce: ~reducer<K, V>, 
        key: K,
        out: Chan<Chan<reduce_proto<V>>>)
    {
        let p = Port();

        send(out, Chan(&p));

        let mut ref_count = 0;
        let mut is_done = false;

        fn get<V: Copy Send>(p: Port<reduce_proto<V>>,
                             ref_count: &mut int, is_done: &mut bool)
           -> Option<V> {
            while !*is_done || *ref_count > 0 {
                match recv(p) {
                  emit_val(v) => {
                    // error!("received %d", v);
                    return Some(v);
                  }
                  done => {
                    // error!("all done");
                    *is_done = true;
                  }
                  addref => { *ref_count += 1; }
                  release => { *ref_count -= 1; }
                }
            }
            return None;
        }

        (*reduce)(&key, || get(p, &mut ref_count, &mut is_done) );
    }

    fn map_reduce<K1: Copy Send, K2: Hash IterBytes Eq Const Copy Send, V: Copy Send>(
        map: mapper<K1, K2, V>,
        reduce: reducer<K2, V>,
        inputs: ~[K1])
    {
        let mut ctrl = ~[];

        // This task becomes the master control task. It task::_spawns
        // to do the rest.

        let reducers = map::HashMap();
        let mut tasks = start_mappers(&map, &mut ctrl, &inputs);
        let mut num_mappers = vec::len(inputs) as int;

        while num_mappers > 0 {
            let (_ready, message, ctrls) = pipes::select(move ctrl);
            match option::unwrap(move message) {
              ctrl_proto::mapper_done => {
                // error!("received mapper terminated.");
                num_mappers -= 1;
                ctrl = move ctrls;
              }
              ctrl_proto::find_reducer(k, cc) => {
                let c;
                // log(error, "finding reducer for " + k);
                match reducers.find(k) {
                  Some(_c) => {
                    // log(error,
                    // "reusing existing reducer for " + k);
                    c = _c;
                  }
                  None => {
                    // log(error, "creating new reducer for " + k);
                    let p = Port();
                    let ch = Chan(&p);
                    let r = reduce, kk = k;
                    tasks.push(spawn_joinable(|move r| reduce_task(~r, kk, ch) ));
                    c = recv(p);
                    reducers.insert(k, c);
                  }
                }
                ctrl = vec::append_one(
                    move ctrls,
                    ctrl_proto::server::reducer(move_out!(cc), c));
              }
            }
        }

        for reducers.each_value |v| { send(v, done) }

        for tasks.each |t| { join(*t); }
    }
}

fn main() {
    let argv = os::args();
    if vec::len(argv) < 2u && !os::getenv(~"RUST_BENCH").is_some() {
        let out = io::stdout();

        out.write_line(fmt!("Usage: %s <filename> ...", argv[0]));

        return;
    }

    let readers: ~[fn~() -> word_reader]  = if argv.len() >= 2 {
        vec::view(argv, 1u, argv.len()).map(|f| {
            let f = *f;
            fn~() -> word_reader { file_word_reader(f) }
        })
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

fn read_word(r: io::Reader) -> Option<~str> {
    let mut w = ~"";

    while !r.eof() {
        let c = r.read_char();

        if is_word_char(c) {
            w += str::from_char(c);
        } else { if w != ~"" { return Some(w); } }
    }
    return None;
}

fn is_word_char(c: char) -> bool {
    char::is_alphabetic(c) || char::is_digit(c) || c == '_'
}

struct random_word_reader {
    mut remaining: uint,
    rng: rand::Rng,
}

impl random_word_reader: word_reader {
    fn read_word() -> Option<~str> {
        if self.remaining > 0 {
            self.remaining -= 1;
            let len = self.rng.gen_uint_range(1, 4);
            Some(self.rng.gen_str(len))
        }
        else { None }
    }
}

fn random_word_reader(count: uint) -> random_word_reader {
    random_word_reader {
        remaining: count,
        rng: rand::Rng()
    }
}
