// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use json;
use sha1;
use serialize::{Encoder, Encodable, Decoder, Decodable};
use sort;

use core::cell::Cell;
use core::cmp;
use core::comm::{PortOne, oneshot, send_one};
use core::either::{Either, Left, Right};
use core::hashmap::HashMap;
use core::io;
use core::pipes::recv;
use core::run;
use core::to_bytes;

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

#[deriving(Eq)]
#[auto_encode]
#[auto_decode]
struct WorkKey {
    kind: ~str,
    name: ~str
}

impl to_bytes::IterBytes for WorkKey {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) {
        let mut flag = true;
        self.kind.iter_bytes(lsb0, |bytes| {flag = f(bytes); flag});
        if !flag { return; }
        self.name.iter_bytes(lsb0, f);
    }
}

impl cmp::Ord for WorkKey {
    fn lt(&self, other: &WorkKey) -> bool {
        self.kind < other.kind ||
            (self.kind == other.kind &&
             self.name < other.name)
    }
    fn le(&self, other: &WorkKey) -> bool {
        self.lt(other) || self.eq(other)
    }
    fn ge(&self, other: &WorkKey) -> bool {
        self.gt(other) || self.eq(other)
    }
    fn gt(&self, other: &WorkKey) -> bool {
        ! self.le(other)
    }
}

pub impl WorkKey {
    fn new(kind: &str, name: &str) -> WorkKey {
    WorkKey { kind: kind.to_owned(), name: name.to_owned() }
    }
}

struct WorkMap(HashMap<WorkKey, ~str>);

impl WorkMap {
    fn new() -> WorkMap { WorkMap(HashMap::new()) }
}

impl<S:Encoder> Encodable<S> for WorkMap {
    fn encode(&self, s: &mut S) {
        let mut d = ~[];
        for self.each |k, v| {
            d.push((copy *k, copy *v))
        }
        sort::tim_sort(d);
        d.encode(s)
    }
}

impl<D:Decoder> Decodable<D> for WorkMap {
    fn decode(d: &mut D) -> WorkMap {
        let v : ~[(WorkKey,~str)] = Decodable::decode(d);
        let mut w = WorkMap::new();
        for v.each |&(k, v)| {
            w.insert(copy k, copy v);
        }
        w
    }
}

struct Database {
    db_filename: Path,
    db_cache: HashMap<~str, ~str>,
    db_dirty: bool
}

pub impl Database {
    fn prepare(&mut self,
               fn_name: &str,
               declared_inputs: &WorkMap)
               -> Option<(WorkMap, WorkMap, ~str)> {
        let k = json_encode(&(fn_name, declared_inputs));
        match self.db_cache.find(&k) {
            None => None,
            Some(v) => Some(json_decode(*v))
        }
    }

    fn cache(&mut self,
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

pub impl Logger {
    fn info(&self, i: &str) {
        io::println(~"workcache: " + i.to_owned());
    }
}

struct Context {
    db: @mut Database,
    logger: @mut Logger,
    cfg: @json::Object,
    freshness: HashMap<~str,@fn(&str,&str)->bool>
}

struct Prep {
    ctxt: @Context,
    fn_name: ~str,
    declared_inputs: WorkMap,
}

struct Exec {
    discovered_inputs: WorkMap,
    discovered_outputs: WorkMap
}

struct Work<T> {
    prep: @mut Prep,
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
        let j = result::unwrap(json::from_reader(rdr));
        let mut decoder = json::Decoder(j);
        Decodable::decode(&mut decoder)
    }
}

fn digest<T:Encodable<json::Encoder>>(t: &T) -> ~str {
    let mut sha = sha1::sha1();
    sha.input_str(json_encode(t));
    sha.result_str()
}

fn digest_file(path: &Path) -> ~str {
    let mut sha = sha1::sha1();
    let s = io::read_whole_file_str(path);
    sha.input_str(*s.get_ref());
    sha.result_str()
}

pub impl Context {

    fn new(db: @mut Database,
                  lg: @mut Logger,
                  cfg: @json::Object) -> Context {
        Context {
            db: db,
            logger: lg,
            cfg: cfg,
            freshness: HashMap::new()
        }
    }

    fn prep<T:Owned +
              Encodable<json::Encoder> +
              Decodable<json::Decoder>>( // FIXME(#5121)
                  @self,
                  fn_name:&str,
                  blk: &fn(@mut Prep)->Work<T>) -> Work<T> {
        let p = @mut Prep {
            ctxt: self,
            fn_name: fn_name.to_owned(),
            declared_inputs: WorkMap::new()
        };
        blk(p)
    }
}


trait TPrep {
    fn declare_input(&mut self, kind:&str, name:&str, val:&str);
    fn is_fresh(&self, cat:&str, kind:&str, name:&str, val:&str) -> bool;
    fn all_fresh(&self, cat:&str, map:&WorkMap) -> bool;
    fn exec<T:Owned +
              Encodable<json::Encoder> +
              Decodable<json::Decoder>>( // FIXME(#5121)
        &self, blk: ~fn(&Exec) -> T) -> Work<T>;
}

impl TPrep for Prep {
    fn declare_input(&mut self, kind:&str, name:&str, val:&str) {
        self.declared_inputs.insert(WorkKey::new(kind, name),
                                 val.to_owned());
    }

    fn is_fresh(&self, cat: &str, kind: &str,
                name: &str, val: &str) -> bool {
        let k = kind.to_owned();
        let f = (*self.ctxt.freshness.get(&k))(name, val);
        let lg = self.ctxt.logger;
            if f {
                lg.info(fmt!("%s %s:%s is fresh",
                             cat, kind, name));
            } else {
                lg.info(fmt!("%s %s:%s is not fresh",
                             cat, kind, name))
            }
        f
    }

    fn all_fresh(&self, cat: &str, map: &WorkMap) -> bool {
        for map.each |k, v| {
            if ! self.is_fresh(cat, k.kind, k.name, *v) {
                return false;
            }
        }
        return true;
    }

    fn exec<T:Owned +
              Encodable<json::Encoder> +
              Decodable<json::Decoder>>( // FIXME(#5121)
            &self, blk: ~fn(&Exec) -> T) -> Work<T> {
        let mut bo = Some(blk);

        let cached = self.ctxt.db.prepare(self.fn_name, &self.declared_inputs);

        match cached {
            Some((ref disc_in, ref disc_out, ref res))
            if self.all_fresh("declared input",
                              &self.declared_inputs) &&
            self.all_fresh("discovered input", disc_in) &&
            self.all_fresh("discovered output", disc_out) => {
                Work::new(@mut copy *self, Left(json_decode(*res)))
            }

            _ => {
                let (port, chan) = oneshot();
                let mut blk = None;
                blk <-> bo;
                let blk = blk.unwrap();
                let chan = Cell(chan);

                do task::spawn {
                    let exe = Exec {
                        discovered_inputs: WorkMap::new(),
                        discovered_outputs: WorkMap::new(),
                    };
                    let chan = chan.take();
                    let v = blk(&exe);
                    send_one(chan, (exe, v));
                }
                Work::new(@mut copy *self, Right(port))
            }
        }
    }
}

pub impl<T:Owned +
         Encodable<json::Encoder> +
         Decodable<json::Decoder>> Work<T> { // FIXME(#5121)
    fn new(p: @mut Prep, e: Either<T,PortOne<(Exec,T)>>) -> Work<T> {
        Work { prep: p, res: Some(e) }
    }
}

// FIXME (#3724): movable self. This should be in impl Work.
fn unwrap<T:Owned +
            Encodable<json::Encoder> +
            Decodable<json::Decoder>>( // FIXME(#5121)
        w: Work<T>) -> T {
    let mut ww = w;
    let mut s = None;

    ww.res <-> s;

    match s {
        None => fail!(),
        Some(Left(v)) => v,
        Some(Right(port)) => {
            let (exe, v) = match recv(port.unwrap()) {
                oneshot::send(data) => data
            };

            let s = json_encode(&v);

            let p = &*ww.prep;
            let db = p.ctxt.db;
            db.cache(p.fn_name,
                 &p.declared_inputs,
                 &exe.discovered_inputs,
                 &exe.discovered_outputs,
                 s);
            v
        }
    }
}

//#[test]
fn test() {
    use core::io::WriterUtil;

    let db = @mut Database { db_filename: Path("db.json"),
                             db_cache: HashMap::new(),
                             db_dirty: false };
    let lg = @mut Logger { a: () };
    let cfg = @HashMap::new();
    let cx = @Context::new(db, lg, cfg);
    let w:Work<~str> = do cx.prep("test1") |prep| {
        let pth = Path("foo.c");
        {
            let file = io::file_writer(&pth, [io::Create]).get();
            file.write_str("int main() { return 0; }");
        }

        prep.declare_input("file", pth.to_str(), digest_file(&pth));
        do prep.exec |_exe| {
            let out = Path("foo.o");
            run::run_program("gcc", [~"foo.c", ~"-o", out.to_str()]);
            out.to_str()
        }
    };
    let s = unwrap(w);
    io::println(s);
}
