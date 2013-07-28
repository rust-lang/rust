// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];


use digest::DigestUtil;
use json;
use sha1::Sha1;
use serialize::{Encoder, Encodable, Decoder, Decodable};
use arc::{Arc,RWArc};
use treemap::TreeMap;

use std::cell::Cell;
use std::comm::{PortOne, oneshot, send_one, recv_one};
use std::either::{Either, Left, Right};
use std::io;
use std::run;
use std::task;

/**
*
* This is a loose clone of the [fbuild build system](https://github.com/felix-lang/fbuild),
* made a touch more generic (not wired to special cases on files) and much
* less metaprogram-y due to rust's comparative weakness there, relative to
* python.
*
* It's based around _imperative builds_ that happen to have some function
* calls cached. That is, it's _just_ a mechanism for describing cached
* functions. This makes it much simpler and smaller than a "build system"
* that produces an IR and evaluates it. The evaluation order is normal
* function calls. Some of them just return really quickly.
*
* A cached function consumes and produces a set of _works_. A work has a
* name, a kind (that determines how the value is to be checked for
* freshness) and a value. Works must also be (de)serializable. Some
* examples of works:
*
*    kind   name    value
*   ------------------------
*    cfg    os      linux
*    file   foo.c   <sha1>
*    url    foo.com <etag>
*
* Works are conceptually single units, but we store them most of the time
* in maps of the form (type,name) => value. These are WorkMaps.
*
* A cached function divides the works it's interested in into inputs and
* outputs, and subdivides those into declared (input) works and
* discovered (input and output) works.
*
* A _declared_ input or is one that is given to the workcache before
* any work actually happens, in the "prep" phase. Even when a function's
* work-doing part (the "exec" phase) never gets called, it has declared
* inputs, which can be checked for freshness (and potentially
* used to determine that the function can be skipped).
*
* The workcache checks _all_ works for freshness, but uses the set of
* discovered outputs from the _previous_ exec (which it will re-discover
* and re-record each time the exec phase runs).
*
* Therefore the discovered works cached in the db might be a
* mis-approximation of the current discoverable works, but this is ok for
* the following reason: we assume that if an artifact A changed from
* depending on B,C,D to depending on B,C,D,E, then A itself changed (as
* part of the change-in-dependencies), so we will be ok.
*
* Each function has a single discriminated output work called its _result_.
* This is only different from other works in that it is returned, by value,
* from a call to the cacheable function; the other output works are used in
* passing to invalidate dependencies elsewhere in the cache, but do not
* otherwise escape from a function invocation. Most functions only have one
* output work anyways.
*
* A database (the central store of a workcache) stores a mappings:
*
* (fn_name,{declared_input}) => ({discovered_input},
*                                {discovered_output},result)
*
* (Note: fbuild, which workcache is based on, has the concept of a declared
* output as separate from a discovered output. This distinction exists only
* as an artifact of how fbuild works: via annotations on function types
* and metaprogramming, with explicit dependency declaration as a fallback.
* Workcache is more explicit about dependencies, and as such treats all
* outputs the same, as discovered-during-the-last-run.)
*
*/

#[deriving(Clone, Eq, Encodable, Decodable, TotalOrd, TotalEq)]
struct WorkKey {
    kind: ~str,
    name: ~str
}

impl WorkKey {
    pub fn new(kind: &str, name: &str) -> WorkKey {
        WorkKey {
            kind: kind.to_owned(),
            name: name.to_owned(),
        }
    }
}

#[deriving(Clone, Eq, Encodable, Decodable)]
struct WorkMap(TreeMap<WorkKey, ~str>);

impl WorkMap {
    fn new() -> WorkMap { WorkMap(TreeMap::new()) }
}

struct Database {
    db_filename: Path,
    db_cache: TreeMap<~str, ~str>,
    db_dirty: bool
}

impl Database {

    pub fn new(p: Path) -> Database {
        Database {
            db_filename: p,
            db_cache: TreeMap::new(),
            db_dirty: false
        }
    }

    pub fn prepare(&self,
                   fn_name: &str,
                   declared_inputs: &WorkMap)
                   -> Option<(WorkMap, WorkMap, ~str)> {
        let k = json_encode(&(fn_name, declared_inputs));
        match self.db_cache.find(&k) {
            None => None,
            Some(v) => Some(json_decode(*v))
        }
    }

    pub fn cache(&mut self,
                 fn_name: &str,
                 declared_inputs: &WorkMap,
                 discovered_inputs: &WorkMap,
                 discovered_outputs: &WorkMap,
                 result: &str) {
        let k = json_encode(&(fn_name, declared_inputs));
        let v = json_encode(&(discovered_inputs,
                              discovered_outputs,
                              result));
        self.db_cache.insert(k,v);
        self.db_dirty = true
    }
}

struct Logger {
    // FIXME #4432: Fill in
    a: ()
}

impl Logger {

    pub fn new() -> Logger {
        Logger { a: () }
    }

    pub fn info(&self, i: &str) {
        io::println(~"workcache: " + i);
    }
}

#[deriving(Clone)]
struct Context {
    db: RWArc<Database>,
    logger: RWArc<Logger>,
    cfg: Arc<json::Object>,
    freshness: Arc<TreeMap<~str,extern fn(&str,&str)->bool>>
}

struct Prep<'self> {
    ctxt: &'self Context,
    fn_name: &'self str,
    declared_inputs: WorkMap,
}

struct Exec {
    discovered_inputs: WorkMap,
    discovered_outputs: WorkMap
}

struct Work<'self, T> {
    prep: &'self Prep<'self>,
    res: Option<Either<T,PortOne<(Exec,T)>>>
}

fn json_encode<T:Encodable<json::Encoder>>(t: &T) -> ~str {
    do io::with_str_writer |wr| {
        let mut encoder = json::Encoder(wr);
        t.encode(&mut encoder);
    }
}

// FIXME(#5121)
fn json_decode<T:Decodable<json::Decoder>>(s: &str) -> T {
    do io::with_str_reader(s) |rdr| {
        let j = json::from_reader(rdr).unwrap();
        let mut decoder = json::Decoder(j);
        Decodable::decode(&mut decoder)
    }
}

fn digest<T:Encodable<json::Encoder>>(t: &T) -> ~str {
    let mut sha = ~Sha1::new();
    (*sha).input_str(json_encode(t));
    (*sha).result_str()
}

fn digest_file(path: &Path) -> ~str {
    let mut sha = ~Sha1::new();
    let s = io::read_whole_file_str(path);
    (*sha).input_str(*s.get_ref());
    (*sha).result_str()
}

impl Context {

    pub fn new(db: RWArc<Database>,
               lg: RWArc<Logger>,
               cfg: Arc<json::Object>) -> Context {
        Context {
            db: db,
            logger: lg,
            cfg: cfg,
            freshness: Arc::new(TreeMap::new())
        }
    }

    pub fn prep<'a>(&'a self, fn_name: &'a str) -> Prep<'a> {
        Prep::new(self, fn_name)
    }

    pub fn with_prep<'a, T>(&'a self, fn_name: &'a str, blk: &fn(p: &mut Prep) -> T) -> T {
        let mut p = self.prep(fn_name);
        blk(&mut p)
    }

}

impl<'self> Prep<'self> {
    fn new(ctxt: &'self Context, fn_name: &'self str) -> Prep<'self> {
        Prep {
            ctxt: ctxt,
            fn_name: fn_name,
            declared_inputs: WorkMap::new()
        }
    }
}

impl<'self> Prep<'self> {
    fn declare_input(&mut self, kind:&str, name:&str, val:&str) {
        self.declared_inputs.insert(WorkKey::new(kind, name),
                                 val.to_owned());
    }

    fn is_fresh(&self, cat: &str, kind: &str,
                name: &str, val: &str) -> bool {
        let k = kind.to_owned();
        let f = self.ctxt.freshness.get().find(&k);
        let fresh = match f {
            None => fail!("missing freshness-function for '%s'", kind),
            Some(f) => (*f)(name, val)
        };
        do self.ctxt.logger.write |lg| {
            if fresh {
                lg.info(fmt!("%s %s:%s is fresh",
                             cat, kind, name));
            } else {
                lg.info(fmt!("%s %s:%s is not fresh",
                             cat, kind, name))
            }
        };
        fresh
    }

    fn all_fresh(&self, cat: &str, map: &WorkMap) -> bool {
        for map.iter().advance |(k, v)| {
            if ! self.is_fresh(cat, k.kind, k.name, *v) {
                return false;
            }
        }
        return true;
    }

    fn exec<T:Send +
        Encodable<json::Encoder> +
        Decodable<json::Decoder>>(
            &'self self, blk: ~fn(&Exec) -> T) -> T {
        self.exec_work(blk).unwrap()
    }

    fn exec_work<T:Send +
        Encodable<json::Encoder> +
        Decodable<json::Decoder>>( // FIXME(#5121)
            &'self self, blk: ~fn(&Exec) -> T) -> Work<'self, T> {
        let mut bo = Some(blk);

        let cached = do self.ctxt.db.read |db| {
            db.prepare(self.fn_name, &self.declared_inputs)
        };

        let res = match cached {
            Some((ref disc_in, ref disc_out, ref res))
            if self.all_fresh("declared input",&self.declared_inputs) &&
               self.all_fresh("discovered input", disc_in) &&
               self.all_fresh("discovered output", disc_out) => {
                Left(json_decode(*res))
            }

            _ => {
                let (port, chan) = oneshot();
                let blk = bo.take_unwrap();
                let chan = Cell::new(chan);

                do task::spawn {
                    let exe = Exec {
                        discovered_inputs: WorkMap::new(),
                        discovered_outputs: WorkMap::new(),
                    };
                    let chan = chan.take();
                    let v = blk(&exe);
                    send_one(chan, (exe, v));
                }
                Right(port)
            }
        };
        Work::new(self, res)
    }
}

impl<'self, T:Send +
       Encodable<json::Encoder> +
       Decodable<json::Decoder>>
    Work<'self, T> { // FIXME(#5121)

    pub fn new(p: &'self Prep<'self>, e: Either<T,PortOne<(Exec,T)>>) -> Work<'self, T> {
        Work { prep: p, res: Some(e) }
    }

    pub fn unwrap(self) -> T {
        let Work { prep, res } = self;
        match res {
            None => fail!(),
            Some(Left(v)) => v,
            Some(Right(port)) => {
                let (exe, v) = recv_one(port);
                let s = json_encode(&v);
                do prep.ctxt.db.write |db| {
                    db.cache(prep.fn_name,
                             &prep.declared_inputs,
                             &exe.discovered_inputs,
                             &exe.discovered_outputs,
                             s);
                }
                v
            }
        }
    }
}


//#[test]
fn test() {
    use std::io::WriterUtil;

    let pth = Path("foo.c");
    {
        let r = io::file_writer(&pth, [io::Create]);
        r.get_ref().write_str("int main() { return 0; }");
    }

    let cx = Context::new(RWArc::new(Database::new(Path("db.json"))),
                          RWArc::new(Logger::new()),
                          Arc::new(TreeMap::new()));

    let s = do cx.with_prep("test1") |prep| {

        let subcx = cx.clone();

        prep.declare_input("file", pth.to_str(), digest_file(&pth));
        do prep.exec |_exe| {
            let out = Path("foo.o");
            run::process_status("gcc", [~"foo.c", ~"-o", out.to_str()]);

            let _proof_of_concept = subcx.prep("subfn");
            // Could run sub-rules inside here.

            out.to_str()
        }
    };
    io::println(s);
}
