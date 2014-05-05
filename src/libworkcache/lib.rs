// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "workcache#0.11.0-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]
#![feature(phase)]
#![allow(visible_private_types)]
#![deny(deprecated_owned_vector)]

#[phase(syntax, link)] extern crate log;
extern crate serialize;
extern crate collections;
extern crate sync;

use serialize::json;
use serialize::json::ToJson;
use serialize::{Encoder, Encodable, Decoder, Decodable};
use sync::{Arc, RWLock};
use collections::TreeMap;
use std::str;
use std::io;
use std::io::{File, MemWriter};

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

#[deriving(Clone, Eq, Encodable, Decodable, Ord, TotalOrd, TotalEq)]
struct WorkKey {
    kind: StrBuf,
    name: StrBuf
}

impl WorkKey {
    pub fn new(kind: &str, name: &str) -> WorkKey {
        WorkKey {
            kind: kind.to_strbuf(),
            name: name.to_strbuf(),
        }
    }
}

// FIXME #8883: The key should be a WorkKey and not a StrBuf.
// This is working around some JSON weirdness.
#[deriving(Clone, Eq, Encodable, Decodable)]
struct WorkMap(TreeMap<StrBuf, KindMap>);

#[deriving(Clone, Eq, Encodable, Decodable)]
struct KindMap(TreeMap<StrBuf, StrBuf>);

impl WorkMap {
    fn new() -> WorkMap { WorkMap(TreeMap::new()) }

    fn insert_work_key(&mut self, k: WorkKey, val: StrBuf) {
        let WorkKey { kind, name } = k;
        let WorkMap(ref mut map) = *self;
        match map.find_mut(&name) {
            Some(&KindMap(ref mut m)) => { m.insert(kind, val); return; }
            None => ()
        }
        let mut new_map = TreeMap::new();
        new_map.insert(kind, val);
        map.insert(name, KindMap(new_map));
    }
}

pub struct Database {
    db_filename: Path,
    db_cache: TreeMap<StrBuf, StrBuf>,
    pub db_dirty: bool,
}

impl Database {

    pub fn new(p: Path) -> Database {
        let mut rslt = Database {
            db_filename: p,
            db_cache: TreeMap::new(),
            db_dirty: false
        };
        if rslt.db_filename.exists() {
            rslt.load();
        }
        rslt
    }

    pub fn prepare(&self,
                   fn_name: &str,
                   declared_inputs: &WorkMap)
                   -> Option<(WorkMap, WorkMap, StrBuf)> {
        let k = json_encode(&(fn_name, declared_inputs));
        match self.db_cache.find(&k) {
            None => None,
            Some(v) => Some(json_decode(v.as_slice()))
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
    fn save(&self) -> io::IoResult<()> {
        let mut f = File::create(&self.db_filename);

        // FIXME(pcwalton): Yuck.
        let mut new_db_cache = TreeMap::new();
        for (ref k, ref v) in self.db_cache.iter() {
            new_db_cache.insert((*k).to_owned(), (*v).to_owned());
        }

        new_db_cache.to_json().to_pretty_writer(&mut f)
    }

    fn load(&mut self) {
        assert!(!self.db_dirty);
        assert!(self.db_filename.exists());
        match File::open(&self.db_filename) {
            Err(e) => fail!("Couldn't load workcache database {}: {}",
                            self.db_filename.display(),
                            e),
            Ok(mut stream) => {
                match json::from_reader(&mut stream) {
                    Err(e) => fail!("Couldn't parse workcache database (from file {}): {}",
                                    self.db_filename.display(), e.to_str()),
                    Ok(r) => {
                        let mut decoder = json::Decoder::new(r);
                        self.db_cache = Decodable::decode(&mut decoder).unwrap();
                    }
                }
            }
        }
    }
}

#[unsafe_destructor]
impl Drop for Database {
    fn drop(&mut self) {
        if self.db_dirty {
            // FIXME: is failing the right thing to do here
            self.save().unwrap();
        }
    }
}

pub type FreshnessMap = TreeMap<StrBuf,extern fn(&str,&str)->bool>;

#[deriving(Clone)]
pub struct Context {
    pub db: Arc<RWLock<Database>>,
    cfg: Arc<json::Object>,
    /// Map from kinds (source, exe, url, etc.) to a freshness function.
    /// The freshness function takes a name (e.g. file path) and value
    /// (e.g. hash of file contents) and determines whether it's up-to-date.
    /// For example, in the file case, this would read the file off disk,
    /// hash it, and return the result of comparing the given hash and the
    /// read hash for equality.
    freshness: Arc<FreshnessMap>
}

pub struct Prep<'a> {
    ctxt: &'a Context,
    fn_name: &'a str,
    declared_inputs: WorkMap,
}

pub struct Exec {
    discovered_inputs: WorkMap,
    discovered_outputs: WorkMap
}

enum Work<'a, T> {
    WorkValue(T),
    WorkFromTask(&'a Prep<'a>, Receiver<(Exec, T)>),
}

fn json_encode<'a, T:Encodable<json::Encoder<'a>, io::IoError>>(t: &T) -> StrBuf {
    let mut writer = MemWriter::new();
    let mut encoder = json::Encoder::new(&mut writer as &mut io::Writer);
    let _ = t.encode(&mut encoder);
    str::from_utf8(writer.unwrap().as_slice()).unwrap().to_strbuf()
}

// FIXME(#5121)
fn json_decode<T:Decodable<json::Decoder, json::DecoderError>>(s: &str) -> T {
    debug!("json decoding: {}", s);
    let j = json::from_str(s).unwrap();
    let mut decoder = json::Decoder::new(j);
    Decodable::decode(&mut decoder).unwrap()
}

impl Context {

    pub fn new(db: Arc<RWLock<Database>>,
               cfg: Arc<json::Object>) -> Context {
        Context::new_with_freshness(db, cfg, Arc::new(TreeMap::new()))
    }

    pub fn new_with_freshness(db: Arc<RWLock<Database>>,
                              cfg: Arc<json::Object>,
                              freshness: Arc<FreshnessMap>) -> Context {
        Context {
            db: db,
            cfg: cfg,
            freshness: freshness
        }
    }

    pub fn prep<'a>(&'a self, fn_name: &'a str) -> Prep<'a> {
        Prep::new(self, fn_name)
    }

    pub fn with_prep<'a,
                     T>(
                     &'a self,
                     fn_name: &'a str,
                     blk: |p: &mut Prep| -> T)
                     -> T {
        let mut p = self.prep(fn_name);
        blk(&mut p)
    }

}

impl Exec {
    pub fn discover_input(&mut self,
                          dependency_kind: &str,
                          dependency_name: &str,
                          dependency_val: &str) {
        debug!("Discovering input {} {} {}", dependency_kind, dependency_name, dependency_val);
        self.discovered_inputs.insert_work_key(WorkKey::new(dependency_kind, dependency_name),
                                 dependency_val.to_strbuf());
    }
    pub fn discover_output(&mut self,
                           dependency_kind: &str,
                           dependency_name: &str,
                           dependency_val: &str) {
        debug!("Discovering output {} {} {}", dependency_kind, dependency_name, dependency_val);
        self.discovered_outputs.insert_work_key(WorkKey::new(dependency_kind, dependency_name),
                                 dependency_val.to_strbuf());
    }

    // returns pairs of (kind, name)
    pub fn lookup_discovered_inputs(&self) -> Vec<(StrBuf, StrBuf)> {
        let mut rs = vec![];
        let WorkMap(ref discovered_inputs) = self.discovered_inputs;
        for (k, v) in discovered_inputs.iter() {
            let KindMap(ref vmap) = *v;
            for (k1, _) in vmap.iter() {
                rs.push((k1.clone(), k.clone()));
            }
        }
        rs
    }
}

impl<'a> Prep<'a> {
    fn new(ctxt: &'a Context, fn_name: &'a str) -> Prep<'a> {
        Prep {
            ctxt: ctxt,
            fn_name: fn_name,
            declared_inputs: WorkMap::new()
        }
    }

    pub fn lookup_declared_inputs(&self) -> Vec<StrBuf> {
        let mut rs = vec![];
        let WorkMap(ref declared_inputs) = self.declared_inputs;
        for (_, v) in declared_inputs.iter() {
            let KindMap(ref vmap) = *v;
            for (inp, _) in vmap.iter() {
                rs.push(inp.clone());
            }
        }
        rs
    }
}

impl<'a> Prep<'a> {
    pub fn declare_input(&mut self, kind: &str, name: &str, val: &str) {
        debug!("Declaring input {} {} {}", kind, name, val);
        self.declared_inputs.insert_work_key(WorkKey::new(kind, name),
                                 val.to_strbuf());
    }

    fn is_fresh(&self, cat: &str, kind: &str, name: &str, val: &str) -> bool {
        let k = kind.to_strbuf();
        let f = self.ctxt.freshness.deref().find(&k);
        debug!("freshness for: {}/{}/{}/{}", cat, kind, name, val)
        let fresh = match f {
            None => fail!("missing freshness-function for '{}'", kind),
            Some(f) => (*f)(name, val)
        };
        if fresh {
            info!("{} {}:{} is fresh", cat, kind, name);
        } else {
            info!("{} {}:{} is not fresh", cat, kind, name);
        }
        fresh
    }

    fn all_fresh(&self, cat: &str, map: &WorkMap) -> bool {
        let WorkMap(ref map) = *map;
        for (k_name, kindmap) in map.iter() {
            let KindMap(ref kindmap_) = *kindmap;
            for (k_kind, v) in kindmap_.iter() {
               if !self.is_fresh(cat,
                                 k_kind.as_slice(),
                                 k_name.as_slice(),
                                 v.as_slice()) {
                  return false;
            }
          }
        }
        return true;
    }

    pub fn exec<'a, T:Send +
        Encodable<json::Encoder<'a>, io::IoError> +
        Decodable<json::Decoder, json::DecoderError>>(
            &'a self, blk: proc(&mut Exec):Send -> T) -> T {
        self.exec_work(blk).unwrap()
    }

    fn exec_work<'a, T:Send +
        Encodable<json::Encoder<'a>, io::IoError> +
        Decodable<json::Decoder, json::DecoderError>>( // FIXME(#5121)
            &'a self, blk: proc(&mut Exec):Send -> T) -> Work<'a, T> {
        let mut bo = Some(blk);

        debug!("exec_work: looking up {} and {:?}", self.fn_name,
               self.declared_inputs);
        let cached = {
            let db = self.ctxt.db.deref().read();
            db.deref().prepare(self.fn_name, &self.declared_inputs)
        };

        match cached {
            Some((ref disc_in, ref disc_out, ref res))
            if self.all_fresh("declared input",&self.declared_inputs) &&
               self.all_fresh("discovered input", disc_in) &&
               self.all_fresh("discovered output", disc_out) => {
                debug!("Cache hit!");
                debug!("Trying to decode: {:?} / {:?} / {}",
                       disc_in, disc_out, *res);
                Work::from_value(json_decode(res.as_slice()))
            }

            _ => {
                debug!("Cache miss!");
                let (tx, rx) = channel();
                let blk = bo.take_unwrap();

                // FIXME: What happens if the task fails?
                spawn(proc() {
                    let mut exe = Exec {
                        discovered_inputs: WorkMap::new(),
                        discovered_outputs: WorkMap::new(),
                    };
                    let v = blk(&mut exe);
                    tx.send((exe, v));
                });
                Work::from_task(self, rx)
            }
        }
    }
}

impl<'a, T:Send +
       Encodable<json::Encoder<'a>, io::IoError> +
       Decodable<json::Decoder, json::DecoderError>>
    Work<'a, T> { // FIXME(#5121)

    pub fn from_value(elt: T) -> Work<'a, T> {
        WorkValue(elt)
    }
    pub fn from_task(prep: &'a Prep<'a>, port: Receiver<(Exec, T)>)
        -> Work<'a, T> {
        WorkFromTask(prep, port)
    }

    pub fn unwrap(self) -> T {
        match self {
            WorkValue(v) => v,
            WorkFromTask(prep, port) => {
                let (exe, v) = port.recv();
                let s = json_encode(&v);
                let mut db = prep.ctxt.db.deref().write();
                db.deref_mut().cache(prep.fn_name,
                                     &prep.declared_inputs,
                                     &exe.discovered_inputs,
                                     &exe.discovered_outputs,
                                     s.as_slice());
                v
            }
        }
    }
}


#[test]
#[cfg(not(target_os="android"))] // FIXME(#10455)
fn test() {
    use std::os;
    use std::io::{fs, Command};
    use std::str::from_utf8;

    // Create a path to a new file 'filename' in the directory in which
    // this test is running.
    fn make_path(filename: StrBuf) -> Path {
        let pth = os::self_exe_path().expect("workcache::test failed").with_filename(filename);
        if pth.exists() {
            fs::unlink(&pth).unwrap();
        }
        return pth;
    }

    let pth = make_path("foo.c".to_strbuf());
    File::create(&pth).write(bytes!("int main() { return 0; }")).unwrap();

    let db_path = make_path("db.json".to_strbuf());

    let cx = Context::new(Arc::new(RWLock::new(Database::new(db_path))),
                          Arc::new(TreeMap::new()));

    let s = cx.with_prep("test1", |prep| {

        let subcx = cx.clone();
        let pth = pth.clone();

        let contents = File::open(&pth).read_to_end().unwrap();
        let file_content = from_utf8(contents.as_slice()).unwrap().to_owned();

        // FIXME (#9639): This needs to handle non-utf8 paths
        prep.declare_input("file", pth.as_str().unwrap(), file_content);
        prep.exec(proc(_exe) {
            let out = make_path("foo.o".to_strbuf());
            let compiler = if cfg!(windows) {"gcc"} else {"cc"};
            Command::new(compiler).arg(pth).arg("-o").arg(out.clone()).status().unwrap();

            let _proof_of_concept = subcx.prep("subfn");
            // Could run sub-rules inside here.

            // FIXME (#9639): This needs to handle non-utf8 paths
            out.as_str().unwrap().to_owned()
        })
    });

    println!("{}", s);
}
