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

use digest::Digest;
use json;
use json::ToJson;
use sha1::Sha1;
use serialize::{Encoder, Encodable, Decoder, Decodable};
use arc::{Arc,RWArc};
use treemap::TreeMap;
use std::cell::Cell;
use std::comm::{PortOne, oneshot};
use std::either::{Either, Left, Right};
use std::{io, os, task};

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

// FIXME #8883: The key should be a WorkKey and not a ~str.
// This is working around some JSON weirdness.
#[deriving(Clone, Eq, Encodable, Decodable)]
struct WorkMap(TreeMap<~str, KindMap>);

#[deriving(Clone, Eq, Encodable, Decodable)]
struct KindMap(TreeMap<~str, ~str>);

impl WorkMap {
    fn new() -> WorkMap { WorkMap(TreeMap::new()) }

    fn insert_work_key(&mut self, k: WorkKey, val: ~str) {
        let WorkKey { kind, name } = k;
        match self.find_mut(&name) {
            Some(&KindMap(ref mut m)) => { m.insert(kind, val); return; }
            None => ()
        }
        let mut new_map = TreeMap::new();
        new_map.insert(kind, val);
        self.insert(name, KindMap(new_map));
    }
}

struct Database {
    db_filename: Path,
    db_cache: TreeMap<~str, ~str>,
    db_dirty: bool
}

impl Database {

    pub fn new(p: Path) -> Database {
        let mut rslt = Database {
            db_filename: p,
            db_cache: TreeMap::new(),
            db_dirty: false
        };
        if os::path_exists(&rslt.db_filename) {
            rslt.load();
        }
        rslt
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

    // FIXME #4330: This should have &mut self and should set self.db_dirty to false.
    fn save(&self) {
        let f = io::file_writer(&self.db_filename, [io::Create, io::Truncate]).unwrap();
        self.db_cache.to_json().to_pretty_writer(f);
    }

    fn load(&mut self) {
        assert!(!self.db_dirty);
        assert!(os::path_exists(&self.db_filename));
        let f = io::file_reader(&self.db_filename);
        match f {
            Err(e) => fail!("Couldn't load workcache database %s: %s",
                            self.db_filename.to_str(), e.to_str()),
            Ok(r) =>
                match json::from_reader(r) {
                    Err(e) => fail!("Couldn't parse workcache database (from file %s): %s",
                                    self.db_filename.to_str(), e.to_str()),
                    Ok(r) => {
                        let mut decoder = json::Decoder(r);
                        self.db_cache = Decodable::decode(&mut decoder);
                    }
            }
        }
    }
}

// FIXME #4330: use &mut self here
#[unsafe_destructor]
impl Drop for Database {
    fn drop(&self) {
        if self.db_dirty {
            self.save();
        }
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

type FreshnessMap = TreeMap<~str,extern fn(&str,&str)->bool>;

#[deriving(Clone)]
struct Context {
    db: RWArc<Database>,
    logger: RWArc<Logger>,
    cfg: Arc<json::Object>,
    /// Map from kinds (source, exe, url, etc.) to a freshness function.
    /// The freshness function takes a name (e.g. file path) and value
    /// (e.g. hash of file contents) and determines whether it's up-to-date.
    /// For example, in the file case, this would read the file off disk,
    /// hash it, and return the result of comparing the given hash and the
    /// read hash for equality.
    freshness: Arc<FreshnessMap>
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
    debug!("json decoding: %s", s);
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
    (*sha).input_str(s.unwrap());
    (*sha).result_str()
}

impl Context {

    pub fn new(db: RWArc<Database>,
               lg: RWArc<Logger>,
               cfg: Arc<json::Object>) -> Context {
        Context::new_with_freshness(db, lg, cfg, Arc::new(TreeMap::new()))
    }

    pub fn new_with_freshness(db: RWArc<Database>,
                              lg: RWArc<Logger>,
                              cfg: Arc<json::Object>,
                              freshness: Arc<FreshnessMap>) -> Context {
        Context {
            db: db,
            logger: lg,
            cfg: cfg,
            freshness: freshness
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

impl Exec {
    pub fn discover_input(&mut self,
                          dependency_kind: &str,
                          dependency_name: &str,
                          dependency_val: &str) {
        debug!("Discovering input %s %s %s", dependency_kind, dependency_name, dependency_val);
        self.discovered_inputs.insert_work_key(WorkKey::new(dependency_kind, dependency_name),
                                 dependency_val.to_owned());
    }
    pub fn discover_output(&mut self,
                           dependency_kind: &str,
                           dependency_name: &str,
                           dependency_val: &str) {
        debug!("Discovering output %s %s %s", dependency_kind, dependency_name, dependency_val);
        self.discovered_outputs.insert_work_key(WorkKey::new(dependency_kind, dependency_name),
                                 dependency_val.to_owned());
    }

    // returns pairs of (kind, name)
    pub fn lookup_discovered_inputs(&self) -> ~[(~str, ~str)] {
        let mut rs = ~[];
        for (k, v) in self.discovered_inputs.iter() {
            for (k1, _) in v.iter() {
                rs.push((k1.clone(), k.clone()));
            }
        }
        rs
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

    pub fn lookup_declared_inputs(&self) -> ~[~str] {
        let mut rs = ~[];
        for (_, v) in self.declared_inputs.iter() {
            for (inp, _) in v.iter() {
                rs.push(inp.clone());
            }
        }
        rs
    }
}

impl<'self> Prep<'self> {
    pub fn declare_input(&mut self, kind: &str, name: &str, val: &str) {
        debug!("Declaring input %s %s %s", kind, name, val);
        self.declared_inputs.insert_work_key(WorkKey::new(kind, name),
                                 val.to_owned());
    }

    fn is_fresh(&self, cat: &str, kind: &str,
                name: &str, val: &str) -> bool {
        let k = kind.to_owned();
        let f = self.ctxt.freshness.get().find(&k);
        debug!("freshness for: %s/%s/%s/%s", cat, kind, name, val)
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
        for (k_name, kindmap) in map.iter() {
            for (k_kind, v) in kindmap.iter() {
               if ! self.is_fresh(cat, *k_kind, *k_name, *v) {
                  return false;
            }
          }
        }
        return true;
    }

    pub fn exec<T:Send +
        Encodable<json::Encoder> +
        Decodable<json::Decoder>>(
            &'self self, blk: ~fn(&mut Exec) -> T) -> T {
        self.exec_work(blk).unwrap()
    }

    fn exec_work<T:Send +
        Encodable<json::Encoder> +
        Decodable<json::Decoder>>( // FIXME(#5121)
            &'self self, blk: ~fn(&mut Exec) -> T) -> Work<'self, T> {
        let mut bo = Some(blk);

        debug!("exec_work: looking up %s and %?", self.fn_name,
               self.declared_inputs);
        let cached = do self.ctxt.db.read |db| {
            db.prepare(self.fn_name, &self.declared_inputs)
        };

        let res = match cached {
            Some((ref disc_in, ref disc_out, ref res))
            if self.all_fresh("declared input",&self.declared_inputs) &&
               self.all_fresh("discovered input", disc_in) &&
               self.all_fresh("discovered output", disc_out) => {
                debug!("Cache hit!");
                debug!("Trying to decode: %? / %? / %?",
                       disc_in, disc_out, *res);
                Left(json_decode(*res))
            }

            _ => {
                debug!("Cache miss!");
                let (port, chan) = oneshot();
                let blk = bo.take_unwrap();
                let chan = Cell::new(chan);

// What happens if the task fails?
                do task::spawn {
                    let mut exe = Exec {
                        discovered_inputs: WorkMap::new(),
                        discovered_outputs: WorkMap::new(),
                    };
                    let chan = chan.take();
                    let v = blk(&mut exe);
                    chan.send((exe, v));
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
                let (exe, v) = port.recv();
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


#[test]
fn test() {
    use std::io::WriterUtil;
    use std::{os, run};

    let pth = Path("foo.c");
    {
        let r = io::file_writer(&pth, [io::Create]);
        r.unwrap().write_str("int main() { return 0; }");
    }

    let db_path = os::self_exe_path().expect("workcache::test failed").pop().push("db.json");
    if os::path_exists(&db_path) {
        os::remove_file(&db_path);
    }

    let cx = Context::new(RWArc::new(Database::new(db_path)),
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
