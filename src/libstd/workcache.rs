// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use core::cmp;
use core::dvec;
use core::either::{Either, Left, Right};
use core::io;
use core::option;
use core::pipes::{recv, oneshot, PortOne, send_one};
use core::prelude::*;
use core::result;
use core::run;
use core::hashmap::linear::LinearMap;
use core::task;
use core::to_bytes;
use core::mutable::Mut;

/**
*
* This is a loose clone of the fbuild build system, made a touch more
* generic (not wired to special cases on files) and much less metaprogram-y
* due to rust's comparative weakness there, relative to python.
*
* It's based around _imperative bulids_ that happen to have some function
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
* A cached function divides the works it's interested up into inputs and
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

#[deriving_eq]
#[auto_encode]
#[auto_decode]
struct WorkKey {
    kind: ~str,
    name: ~str
}

impl WorkKey: to_bytes::IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) {
        let mut flag = true;
        self.kind.iter_bytes(lsb0, |bytes| {flag = f(bytes); flag});
        if !flag { return; }
        self.name.iter_bytes(lsb0, f);
    }
}

impl WorkKey: cmp::Ord {
    pure fn lt(&self, other: &WorkKey) -> bool {
        self.kind < other.kind ||
            (self.kind == other.kind &&
             self.name < other.name)
    }
    pure fn le(&self, other: &WorkKey) -> bool {
        self.lt(other) || self.eq(other)
    }
    pure fn ge(&self, other: &WorkKey) -> bool {
        self.gt(other) || self.eq(other)
    }
    pure fn gt(&self, other: &WorkKey) -> bool {
        ! self.le(other)
    }
}

impl WorkKey {
    static fn new(kind: &str, name: &str) -> WorkKey {
    WorkKey { kind: kind.to_owned(), name: name.to_owned() }
    }
}

type WorkMap = LinearMap<WorkKey, ~str>;

pub impl<S: Encoder> WorkMap: Encodable<S> {
    fn encode(&self, s: &S) {
        let d = dvec::DVec();
        for self.each |k, v| {
            d.push((copy *k, copy *v))
        }
        let mut v = d.get();
        sort::tim_sort(v);
        v.encode(s)
    }
}

pub impl<D: Decoder> WorkMap: Decodable<D> {
    static fn decode(&self, d: &D) -> WorkMap {
        let v : ~[(WorkKey,~str)] = Decodable::decode(d);
        let mut w = LinearMap();
        for v.each |&(k,v)| {
            w.insert(copy k, copy v);
        }
        w
    }
}

struct Database {
    db_filename: Path,
    db_cache: LinearMap<~str, ~str>,
    mut db_dirty: bool
}

impl Database {

    fn prepare(&mut self,
               fn_name: &str,
               declared_inputs: &WorkMap) ->
        Option<(WorkMap, WorkMap, ~str)> {
        let k = json_encode(&(fn_name, declared_inputs));
        match self.db_cache.find_copy(&k) {
            None => None,
            Some(v) => Some(json_decode(v))
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

impl Logger {
    fn info(i: &str) {
        io::println(~"workcache: " + i.to_owned());
    }
}

struct Context {
    db: @Mut<Database>,
    logger: @Mut<Logger>,
    cfg: @json::Object,
    freshness: LinearMap<~str,@fn(&str,&str)->bool>
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

struct Work<T:Owned> {
    prep: @Mut<Prep>,
    res: Option<Either<T,PortOne<(Exec,T)>>>
}

fn json_encode<T:Encodable<json::Encoder>>(t: &T) -> ~str {
    do io::with_str_writer |wr| {
        t.encode(&json::Encoder(wr));
    }
}

fn json_decode<T:Decodable<json::Decoder>>(s: &str) -> T {
    do io::with_str_reader(s) |rdr| {
        let j = result::unwrap(json::from_reader(rdr));
        Decodable::decode(&json::Decoder(move j))
    }
}

fn digest<T:Encodable<json::Encoder>>(t: &T) -> ~str {
    let sha = sha1::sha1();
    sha.input_str(json_encode(t));
    sha.result_str()
}

fn digest_file(path: &Path) -> ~str {
    let sha = sha1::sha1();
    let s = io::read_whole_file_str(path);
    sha.input_str(*s.get_ref());
    sha.result_str()
}

impl Context {

    static fn new(db: @Mut<Database>,
                  lg: @Mut<Logger>,
                  cfg: @json::Object) -> Context {
        Context {db: db, logger: lg, cfg: cfg, freshness: LinearMap()}
    }

    fn prep<T:Owned
              Encodable<json::Encoder>
              Decodable<json::Decoder>>(
                  @self,
                  fn_name:&str,
                  blk: fn(@Mut<Prep>)->Work<T>) -> Work<T> {
        let p = @Mut(Prep {ctxt: self,
                           fn_name: fn_name.to_owned(),
                           declared_inputs: LinearMap()});
        blk(p)
    }
}


trait TPrep {
    fn declare_input(&self, kind:&str, name:&str, val:&str);
    fn is_fresh(&self, cat:&str, kind:&str, name:&str, val:&str) -> bool;
    fn all_fresh(&self, cat:&str, map:&WorkMap) -> bool;
    fn exec<T:Owned
        Encodable<json::Encoder>
        Decodable<json::Decoder>>(&self, blk: ~fn(&Exec) -> T) -> Work<T>;
}

impl @Mut<Prep> : TPrep {
    fn declare_input(&self, kind:&str, name:&str, val:&str) {
        do self.borrow_mut |p| {
            p.declared_inputs.insert(WorkKey::new(kind, name),
                                     val.to_owned());
        }
    }

    fn is_fresh(&self, cat: &str, kind: &str,
                name: &str, val: &str) -> bool {
        do self.borrow_imm |p| {
            let k = kind.to_owned();
            let f = (*p.ctxt.freshness.get(&k))(name, val);
            do p.ctxt.logger.borrow_imm |lg| {
                if f {
                    lg.info(fmt!("%s %s:%s is fresh",
                                 cat, kind, name));
                } else {
                    lg.info(fmt!("%s %s:%s is not fresh",
                                 cat, kind, name))
                }
            }
            f
        }
    }

    fn all_fresh(&self, cat: &str, map: &WorkMap) -> bool {
        for map.each |k,v| {
            if ! self.is_fresh(cat, k.kind, k.name, *v) {
                return false;
            }
        }
        return true;
    }

    fn exec<T:Owned
        Encodable<json::Encoder>
        Decodable<json::Decoder>>(&self,
                                  blk: ~fn(&Exec) -> T) -> Work<T> {

        let mut bo = Some(move blk);

        do self.borrow_imm |p| {
            let cached = do p.ctxt.db.borrow_mut |db| {
                db.prepare(p.fn_name, &p.declared_inputs)
            };

            match move cached {
                Some((ref disc_in, ref disc_out, ref res))
                if self.all_fresh("declared input",
                                  &p.declared_inputs) &&
                self.all_fresh("discovered input", disc_in) &&
                self.all_fresh("discovered output", disc_out) => {
                    Work::new(*self, move Left(json_decode(*res)))
                }

                _ => {
                    let (chan, port) = oneshot::init();
                    let mut blk = None;
                    blk <-> bo;
                    let blk = blk.unwrap();
                    let chan = ~mut Some(move chan);
                    do task::spawn |move blk, move chan| {
                        let exe = Exec { discovered_inputs: LinearMap(),
                                         discovered_outputs: LinearMap() };
                        let chan = option::swap_unwrap(&mut *chan);
                        let v = blk(&exe);
                        send_one(move chan, (move exe, move v));
                    }

                    Work::new(*self, move Right(move port))
                }
            }
        }
    }
}

impl<T:Owned
       Encodable<json::Encoder>
       Decodable<json::Decoder>>
    Work<T> {
    static fn new(p: @Mut<Prep>, e: Either<T,PortOne<(Exec,T)>>) -> Work<T> {
        move Work { prep: p, res: Some(move e) }
    }
}

// FIXME (#3724): movable self. This should be in impl Work.
fn unwrap<T:Owned
            Encodable<json::Encoder>
            Decodable<json::Decoder>>(w: Work<T>) -> T {

    let mut ww = move w;
    let mut s = None;

    ww.res <-> s;

    match move s {
        None => fail,
        Some(Left(move v)) => move v,
        Some(Right(move port)) => {

            let (exe, v) = match recv(move port) {
                oneshot::send(move data) => move data
            };

            let s = json_encode(&v);

            do ww.prep.borrow_imm |p| {
                do p.ctxt.db.borrow_mut |db| {
                    db.cache(p.fn_name,
                             &p.declared_inputs,
                             &exe.discovered_inputs,
                             &exe.discovered_outputs,
                             s);
                }
            }
            move v
        }
    }
}

//#[test]
fn test() {
    use io::WriterUtil;

    let db = @Mut(Database { db_filename: Path("db.json"),
                             db_cache: LinearMap(),
                             db_dirty: false });
    let lg = @Mut(Logger { a: () });
    let cfg = @LinearMap();
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
            move out.to_str()
        }
    };
    let s = unwrap(move w);
    io::println(s);
}
