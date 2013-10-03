// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;
use std::comm::{SharedPort, SharedChan};
use std::comm;
use std::fmt;
use std::hashmap::{HashMap, HashSet};
use std::local_data;
use std::rt::io::buffered::BufferedWriter;
use std::rt::io::file::{FileInfo, DirectoryInfo};
use std::rt::io::file;
use std::rt::io;
use std::rt::io::Reader;
use std::str;
use std::task;
use std::unstable::finally::Finally;
use std::util;
use std::vec;

use extra::arc::RWArc;
use extra::json::ToJson;
use extra::sort;

use syntax::ast;
use syntax::attr;

use clean;
use doctree;
use fold::DocFolder;
use html::escape::Escape;
use html::format::{VisSpace, Method, PuritySpace};
use html::layout;
use html::markdown::Markdown;

#[deriving(Clone)]
pub struct Context {
    current: ~[~str],
    root_path: ~str,
    dst: Path,
    layout: layout::Layout,
    sidebar: HashMap<~str, ~[~str]>,
    include_sources: bool,
}

pub enum ExternalLocation {
    Remote(~str),   // remote url root of the documentation
    Local,          // inside local folder
    Unknown,        // unknown where the documentation is
}

enum Implementor {
    PathType(clean::Type),
    OtherType(clean::Generics, /* trait */ clean::Type, /* for */ clean::Type),
}

struct Cache {
    // typaram id => name of that typaram
    typarams: HashMap<ast::NodeId, ~str>,
    // type id => all implementations for that type
    impls: HashMap<ast::NodeId, ~[(clean::Impl, Option<~str>)]>,
    // path id => (full qualified path, shortty) -- used to generate urls
    paths: HashMap<ast::NodeId, (~[~str], &'static str)>,
    // trait id => method name => dox
    traits: HashMap<ast::NodeId, HashMap<~str, ~str>>,
    // trait id => implementors of the trait
    implementors: HashMap<ast::NodeId, ~[Implementor]>,
    // crate number => where is the crate's dox located at
    extern_locations: HashMap<ast::CrateNum, ExternalLocation>,

    priv stack: ~[~str],
    priv parent_stack: ~[ast::NodeId],
    priv search_index: ~[IndexItem],
}

struct SourceCollector<'self> {
    seen: HashSet<~str>,
    dst: Path,
    cx: &'self mut Context,
}

struct Item<'self> { cx: &'self Context, item: &'self clean::Item, }
struct Sidebar<'self> { cx: &'self Context, item: &'self clean::Item, }

struct IndexItem {
    ty: &'static str,
    name: ~str,
    path: ~str,
    desc: ~str,
    parent: Option<ast::NodeId>,
}

struct Source<'self>(&'self str);

local_data_key!(pub cache_key: RWArc<Cache>)
local_data_key!(pub current_location_key: ~[~str])

/// Generates the documentation for `crate` into the directory `dst`
pub fn run(mut crate: clean::Crate, dst: Path) {
    let mut cx = Context {
        dst: dst,
        current: ~[],
        root_path: ~"",
        sidebar: HashMap::new(),
        layout: layout::Layout {
            logo: ~"",
            favicon: ~"",
            crate: crate.name.clone(),
        },
        include_sources: true,
    };
    mkdir(&cx.dst);

    match crate.module.get_ref().doc_list() {
        Some(attrs) => {
            for attr in attrs.iter() {
                match *attr {
                    clean::NameValue(~"html_favicon_url", ref s) => {
                        cx.layout.favicon = s.to_owned();
                    }
                    clean::NameValue(~"html_logo_url", ref s) => {
                        cx.layout.logo = s.to_owned();
                    }
                    clean::Word(~"html_no_source") => {
                        cx.include_sources = false;
                    }
                    _ => {}
                }
            }
        }
        None => {}
    }

    // Crawl the crate to build various caches used for the output
    let mut cache = Cache {
        impls: HashMap::new(),
        typarams: HashMap::new(),
        paths: HashMap::new(),
        traits: HashMap::new(),
        implementors: HashMap::new(),
        stack: ~[],
        parent_stack: ~[],
        search_index: ~[],
        extern_locations: HashMap::new(),
    };
    cache.stack.push(crate.name.clone());
    crate = cache.fold_crate(crate);

    // Add all the static files
    let dst = cx.dst.push(crate.name);
    mkdir(&dst);
    write(dst.push("jquery.js"), include_str!("static/jquery-2.0.3.min.js"));
    write(dst.push("main.js"), include_str!("static/main.js"));
    write(dst.push("main.css"), include_str!("static/main.css"));
    write(dst.push("normalize.css"), include_str!("static/normalize.css"));

    // Publish the search index
    {
        let dst = dst.push("search-index.js");
        let mut w = BufferedWriter::new(dst.open_writer(io::CreateOrTruncate));
        let w = &mut w as &mut io::Writer;
        write!(w, "var searchIndex = [");
        for (i, item) in cache.search_index.iter().enumerate() {
            if i > 0 { write!(w, ","); }
            write!(w, "\\{ty:\"{}\",name:\"{}\",path:\"{}\",desc:{}",
                   item.ty, item.name, item.path,
                   item.desc.to_json().to_str())
            match item.parent {
                Some(id) => { write!(w, ",parent:'{}'", id); }
                None => {}
            }
            write!(w, "\\}");
        }
        write!(w, "];");
        write!(w, "var allPaths = \\{");
        for (i, (&id, &(ref fqp, short))) in cache.paths.iter().enumerate() {
            if i > 0 { write!(w, ","); }
            write!(w, "'{}':\\{type:'{}',name:'{}'\\}", id, short, *fqp.last());
        }
        write!(w, "\\};");
        w.flush();
    }

    // Render all source files (this may turn into a giant no-op)
    {
        info2!("emitting source files");
        let dst = cx.dst.push("src");
        mkdir(&dst);
        let dst = dst.push(crate.name);
        mkdir(&dst);
        let mut folder = SourceCollector {
            dst: dst,
            seen: HashSet::new(),
            cx: &mut cx,
        };
        crate = folder.fold_crate(crate);
    }

    for (&n, e) in crate.externs.iter() {
        cache.extern_locations.insert(n, extern_location(e, &cx.dst));
    }

    // And finally render the whole crate's documentation
    cx.crate(crate, cache);
}

fn write(dst: Path, contents: &str) {
    let mut w = dst.open_writer(io::CreateOrTruncate);
    w.write(contents.as_bytes());
}

fn mkdir(path: &Path) {
    do io::io_error::cond.trap(|err| {
        error2!("Couldn't create directory `{}`: {}",
                path.to_str(), err.desc);
        fail2!()
    }).inside {
        if !path.is_dir() {
            file::mkdir(path);
        }
    }
}

fn clean_srcpath(src: &str, f: &fn(&str)) {
    let p = Path(src);
    for c in p.components.iter() {
        if "." == *c {
            continue
        }
        if ".." == *c {
            f("up");
        } else {
            f(c.as_slice())
        }
    }
}

fn extern_location(e: &clean::ExternalCrate, dst: &Path) -> ExternalLocation {
    // See if there's documentation generated into the local directory
    let local_location = dst.push(e.name);
    if local_location.is_dir() {
        return Local;
    }

    // Failing that, see if there's an attribute specifying where to find this
    // external crate
    for attr in e.attrs.iter() {
        match *attr {
            clean::List(~"doc", ref list) => {
                for attr in list.iter() {
                    match *attr {
                        clean::NameValue(~"html_root_url", ref s) => {
                            if s.ends_with("/") {
                                return Remote(s.to_owned());
                            }
                            return Remote(*s + "/");
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Well, at least we tried.
    return Unknown;
}

impl<'self> DocFolder for SourceCollector<'self> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if self.cx.include_sources && !self.seen.contains(&item.source.filename) {
            // If it turns out that we couldn't read this file, then we probably
            // can't read any of the files (generating html output from json or
            // something like that), so just don't include sources for the
            // entire crate. The other option is maintaining this mapping on a
            // per-file basis, but that's probably not worth it...
            self.cx.include_sources = self.emit_source(item.source.filename);
            self.seen.insert(item.source.filename.clone());

            if !self.cx.include_sources {
                println!("warning: source code was requested to be rendered, \
                          but `{}` is a missing source file.",
                         item.source.filename);
                println!("         skipping rendering of source code");
            }
        }
        self.fold_item_recur(item)
    }
}

impl<'self> SourceCollector<'self> {
    fn emit_source(&mut self, filename: &str) -> bool {
        let p = Path(filename);

        // Read the contents of the file
        let mut contents = ~[];
        {
            let mut buf = [0, ..1024];
            let r = do io::io_error::cond.trap(|_| {}).inside {
                p.open_reader(io::Open)
            };
            // If we couldn't open this file, then just returns because it
            // probably means that it's some standard library macro thing and we
            // can't have the source to it anyway.
            let mut r = match r {
                Some(r) => r,
                // eew macro hacks
                None => return filename == "<std-macros>"
            };

            // read everything
            loop {
                match r.read(buf) {
                    Some(n) => contents.push_all(buf.slice_to(n)),
                    None => break
                }
            }
        }
        let contents = str::from_utf8_owned(contents);

        // Create the intermediate directories
        let mut cur = self.dst.clone();
        let mut root_path = ~"../../";
        do clean_srcpath(p.pop().to_str()) |component| {
            cur = cur.push(component);
            mkdir(&cur);
            root_path.push_str("../");
        }

        let dst = cur.push(*p.components.last() + ".html");
        let w = dst.open_writer(io::CreateOrTruncate);
        let mut w = BufferedWriter::new(w);

        let title = format!("{} -- source", *dst.components.last());
        let page = layout::Page {
            title: title,
            ty: "source",
            root_path: root_path,
        };
        layout::render(&mut w as &mut io::Writer, &self.cx.layout,
                       &page, &(""), &Source(contents.as_slice()));
        w.flush();
        return true;
    }
}

impl DocFolder for Cache {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        // Register any generics to their corresponding string. This is used
        // when pretty-printing types
        match item.inner {
            clean::StructItem(ref s)   => self.generics(&s.generics),
            clean::EnumItem(ref e)     => self.generics(&e.generics),
            clean::FunctionItem(ref f) => self.generics(&f.generics),
            clean::TypedefItem(ref t)  => self.generics(&t.generics),
            clean::TraitItem(ref t)    => self.generics(&t.generics),
            clean::ImplItem(ref i)     => self.generics(&i.generics),
            clean::TyMethodItem(ref i) => self.generics(&i.generics),
            clean::MethodItem(ref i)   => self.generics(&i.generics),
            _ => {}
        }

        // Propagate a trait methods' documentation to all implementors of the
        // trait
        match item.inner {
            clean::TraitItem(ref t) => {
                let mut dox = HashMap::new();
                for meth in t.methods.iter() {
                    let it = meth.item();
                    match it.doc_value() {
                        None => {}
                        Some(s) => {
                            dox.insert(it.name.get_ref().to_owned(),
                                       s.to_owned());
                        }
                    }
                }
                self.traits.insert(item.id, dox);
            }
            _ => {}
        }

        // Collect all the implementors of traits.
        match item.inner {
            clean::ImplItem(ref i) => {
                match i.trait_ {
                    Some(clean::ResolvedPath{ id, _ }) => {
                        let v = do self.implementors.find_or_insert_with(id) |_|{
                            ~[]
                        };
                        match i.for_ {
                            clean::ResolvedPath{_} => {
                                v.unshift(PathType(i.for_.clone()));
                            }
                            _ => {
                                v.push(OtherType(i.generics.clone(),
                                                 i.trait_.get_ref().clone(),
                                                 i.for_.clone()));
                            }
                        }
                    }
                    Some(*) | None => {}
                }
            }
            _ => {}
        }

        // Index this method for searching later on
        match item.name {
            Some(ref s) => {
                let parent = match item.inner {
                    clean::TyMethodItem(*) |
                    clean::StructFieldItem(*) |
                    clean::VariantItem(*) => {
                        Some((Some(*self.parent_stack.last()),
                              self.stack.slice_to(self.stack.len() - 1)))

                    }
                    clean::MethodItem(*) => {
                        if self.parent_stack.len() == 0 {
                            None
                        } else {
                            let last = self.parent_stack.last();
                            let amt = match self.paths.find(last) {
                                Some(&(_, "trait")) => self.stack.len() - 1,
                                Some(*) | None => self.stack.len(),
                            };
                            Some((Some(*last), self.stack.slice_to(amt)))
                        }
                    }
                    _ => Some((None, self.stack.as_slice()))
                };
                match parent {
                    Some((parent, path)) => {
                        self.search_index.push(IndexItem {
                            ty: shortty(&item),
                            name: s.to_owned(),
                            path: path.connect("::"),
                            desc: shorter(item.doc_value()).to_owned(),
                            parent: parent,
                        });
                    }
                    None => {}
                }
            }
            None => {}
        }

        // Keep track of the fully qualified path for this item.
        let pushed = if item.name.is_some() {
            let n = item.name.get_ref();
            if n.len() > 0 {
                self.stack.push(n.to_owned());
                true
            } else { false }
        } else { false };
        match item.inner {
            clean::StructItem(*) | clean::EnumItem(*) |
            clean::TypedefItem(*) | clean::TraitItem(*) |
            clean::FunctionItem(*) | clean::ModuleItem(*) |
            clean::VariantItem(*) => {
                self.paths.insert(item.id, (self.stack.clone(), shortty(&item)));
            }
            _ => {}
        }

        // Maintain the parent stack
        let parent_pushed = match item.inner {
            clean::TraitItem(*) | clean::EnumItem(*) | clean::StructItem(*) => {
                self.parent_stack.push(item.id); true
            }
            clean::ImplItem(ref i) => {
                match i.for_ {
                    clean::ResolvedPath{ id, _ } => {
                        self.parent_stack.push(id); true
                    }
                    _ => false
                }
            }
            _ => false
        };

        // Once we've recursively found all the generics, then hoard off all the
        // implementations elsewhere
        let ret = match self.fold_item_recur(item) {
            Some(item) => {
                match item {
                    clean::Item{ attrs, inner: clean::ImplItem(i), _ } => {
                        match i.for_ {
                            clean::ResolvedPath { id, _ } => {
                                let v = do self.impls.find_or_insert_with(id) |_| {
                                    ~[]
                                };
                                // extract relevant documentation for this impl
                                match attrs.move_iter().find(|a| {
                                    match *a {
                                        clean::NameValue(~"doc", _) => true,
                                        _ => false
                                    }
                                }) {
                                    Some(clean::NameValue(_, dox)) => {
                                        v.push((i, Some(dox)));
                                    }
                                    Some(*) | None => {
                                        v.push((i, None));
                                    }
                                }
                            }
                            _ => {}
                        }
                        None
                    }
                    i => Some(i),
                }
            }
            i => i,
        };

        if pushed { self.stack.pop(); }
        if parent_pushed { self.parent_stack.pop(); }
        return ret;
    }
}

impl<'self> Cache {
    fn generics(&mut self, generics: &clean::Generics) {
        for typ in generics.type_params.iter() {
            self.typarams.insert(typ.id, typ.name.clone());
        }
    }
}

impl Context {
    fn recurse<T>(&mut self, s: ~str, f: &fn(&mut Context) -> T) -> T {
        // Recurse in the directory structure and change the "root path" to make
        // sure it always points to the top (relatively)
        if s.len() == 0 {
            fail2!("what {:?}", self);
        }
        let next = self.dst.push(s);
        let prev = util::replace(&mut self.dst, next);
        self.root_path.push_str("../");
        self.current.push(s);

        mkdir(&self.dst);
        let ret = f(self);

        // Go back to where we were at
        self.dst = prev;
        let len = self.root_path.len();
        self.root_path.truncate(len - 3);
        self.current.pop();

        return ret;
    }

    fn crate(self, mut crate: clean::Crate, cache: Cache) {
        enum Work {
            Die,
            Process(Context, clean::Item),
        }
        enum Progress { JobNew, JobDone }
        static WORKERS: int = 10;

        let mut item = match crate.module.take() {
            Some(i) => i,
            None => return
        };
        item.name = Some(crate.name);

        let (port, chan) = comm::stream::<Work>();
        let port = SharedPort::new(port);
        let chan = SharedChan::new(chan);
        let (prog_port, prog_chan) = comm::stream();
        let prog_chan = SharedChan::new(prog_chan);
        let cache = RWArc::new(cache);

        for i in range(0, WORKERS) {
            let port = port.clone();
            let chan = chan.clone();
            let prog_chan = prog_chan.clone();

            let mut task = task::task();
            task.unlinked(); // we kill things manually
            task.name(format!("worker{}", i));
            task.spawn_with(cache.clone(),
                            |cache| worker(cache, &port, &chan, &prog_chan));

            fn worker(cache: RWArc<Cache>,
                      port: &SharedPort<Work>,
                      chan: &SharedChan<Work>,
                      prog_chan: &SharedChan<Progress>) {
                #[fixed_stack_segment]; // we hit markdown FFI *a lot*
                local_data::set(cache_key, cache);

                loop {
                    match port.recv() {
                        Process(cx, item) => {
                            let mut cx = cx;
                            let item = Cell::new(item);
                            do (|| {
                                do cx.item(item.take()) |cx, item| {
                                    prog_chan.send(JobNew);
                                    chan.send(Process(cx.clone(), item));
                                }
                            }).finally {
                                // If we fail, everything else should still get
                                // completed
                                prog_chan.send(JobDone);
                            }
                        }
                        Die => break,
                    }
                }
            }
        }

        chan.send(Process(self, item));
        let mut jobs = 1;
        loop {
            match prog_port.recv() {
                JobNew => jobs += 1,
                JobDone => jobs -= 1,
            }

            if jobs == 0 { break }
        }

        for _ in range(0, WORKERS) {
            chan.send(Die);
        }
    }

    fn item(&mut self, item: clean::Item, f: &fn(&mut Context, clean::Item)) {
        fn render(w: io::file::FileWriter, cx: &mut Context, it: &clean::Item,
                  pushname: bool) {
            // A little unfortunate that this is done like this, but it sure
            // does make formatting *a lot* nicer.
            local_data::set(current_location_key, cx.current.clone());

            let mut title = cx.current.connect("::");
            if pushname {
                if title.len() > 0 { title.push_str("::"); }
                title.push_str(*it.name.get_ref());
            }
            title.push_str(" - Rust");
            let page = layout::Page {
                ty: shortty(it),
                root_path: cx.root_path,
                title: title,
            };

            // We have a huge number of calls to write, so try to alleviate some
            // of the pain by using a buffered writer instead of invoking the
            // write sycall all the time.
            let mut writer = BufferedWriter::new(w);
            layout::render(&mut writer as &mut io::Writer, &cx.layout, &page,
                           &Sidebar{ cx: cx, item: it },
                           &Item{ cx: cx, item: it });
            writer.flush();
        }

        match item.inner {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            clean::ModuleItem(*) => {
                let name = item.name.get_ref().to_owned();
                let item = Cell::new(item);
                do self.recurse(name) |this| {
                    let item = item.take();
                    let dst = this.dst.push("index.html");
                    let writer = dst.open_writer(io::CreateOrTruncate);
                    render(writer.unwrap(), this, &item, false);

                    let m = match item.inner {
                        clean::ModuleItem(m) => m,
                        _ => unreachable!()
                    };
                    this.sidebar = build_sidebar(&m);
                    for item in m.items.move_iter() {
                        f(this, item);
                    }
                }
            }

            // Things which don't have names (like impls) don't get special
            // pages dedicated to them.
            _ if item.name.is_some() => {
                let dst = self.dst.push(item_path(&item));
                let writer = dst.open_writer(io::CreateOrTruncate);
                render(writer.unwrap(), self, &item, true);
            }

            _ => {}
        }
    }
}

fn shortty(item: &clean::Item) -> &'static str {
    match item.inner {
        clean::ModuleItem(*)          => "mod",
        clean::StructItem(*)          => "struct",
        clean::EnumItem(*)            => "enum",
        clean::FunctionItem(*)        => "fn",
        clean::TypedefItem(*)         => "typedef",
        clean::StaticItem(*)          => "static",
        clean::TraitItem(*)           => "trait",
        clean::ImplItem(*)            => "impl",
        clean::ViewItemItem(*)        => "viewitem",
        clean::TyMethodItem(*)        => "tymethod",
        clean::MethodItem(*)          => "method",
        clean::StructFieldItem(*)     => "structfield",
        clean::VariantItem(*)         => "variant",
        clean::ForeignFunctionItem(*) => "ffi",
        clean::ForeignStaticItem(*)   => "ffs",
    }
}

impl<'self> Item<'self> {
    fn ismodule(&self) -> bool {
        match self.item.inner {
            clean::ModuleItem(*) => true, _ => false
        }
    }
}

impl<'self> fmt::Default for Item<'self> {
    fn fmt(it: &Item<'self>, fmt: &mut fmt::Formatter) {
        match attr::find_stability(it.item.attrs.iter()) {
            Some(stability) => {
                write!(fmt.buf,
                       "<a class='stability {lvl}' title='{reason}'>{lvl}</a>",
                       lvl = stability.level.to_str(),
                       reason = match stability.text {
                           Some(s) => s, None => @"",
                       });
            }
            None => {}
        }

        if it.cx.include_sources {
            let mut path = ~[];
            do clean_srcpath(it.item.source.filename) |component| {
                path.push(component.to_owned());
            }
            let href = if it.item.source.loline == it.item.source.hiline {
                format!("{}", it.item.source.loline)
            } else {
                format!("{}-{}", it.item.source.loline, it.item.source.hiline)
            };
            write!(fmt.buf,
                   "<a class='source'
                       href='{root}src/{crate}/{path}.html\\#{href}'>[src]</a>",
                   root = it.cx.root_path,
                   crate = it.cx.layout.crate,
                   path = path.connect("/"),
                   href = href);
        }

        // Write the breadcrumb trail header for the top
        write!(fmt.buf, "<h1 class='fqn'>");
        match it.item.inner {
            clean::ModuleItem(*) => write!(fmt.buf, "Module "),
            clean::FunctionItem(*) => write!(fmt.buf, "Function "),
            clean::TraitItem(*) => write!(fmt.buf, "Trait "),
            clean::StructItem(*) => write!(fmt.buf, "Struct "),
            clean::EnumItem(*) => write!(fmt.buf, "Enum "),
            _ => {}
        }
        let cur = it.cx.current.as_slice();
        let amt = if it.ismodule() { cur.len() - 1 } else { cur.len() };
        for (i, component) in cur.iter().enumerate().take(amt) {
            let mut trail = ~"";
            for _ in range(0, cur.len() - i - 1) {
                trail.push_str("../");
            }
            write!(fmt.buf, "<a href='{}index.html'>{}</a>::",
                   trail, component.as_slice());
        }
        write!(fmt.buf, "<a class='{}' href=''>{}</a></h1>",
               shortty(it.item), it.item.name.get_ref().as_slice());

        match it.item.inner {
            clean::ModuleItem(ref m) => item_module(fmt.buf, it.cx,
                                                    it.item, m.items),
            clean::FunctionItem(ref f) | clean::ForeignFunctionItem(ref f) =>
                item_function(fmt.buf, it.item, f),
            clean::TraitItem(ref t) => item_trait(fmt.buf, it.item, t),
            clean::StructItem(ref s) => item_struct(fmt.buf, it.item, s),
            clean::EnumItem(ref e) => item_enum(fmt.buf, it.item, e),
            clean::TypedefItem(ref t) => item_typedef(fmt.buf, it.item, t),
            _ => {}
        }
    }
}

fn item_path(item: &clean::Item) -> ~str {
    match item.inner {
        clean::ModuleItem(*) => *item.name.get_ref() + "/index.html",
        _ => shortty(item) + "." + *item.name.get_ref() + ".html"
    }
}

fn full_path(cx: &Context, item: &clean::Item) -> ~str {
    let mut s = cx.current.connect("::");
    s.push_str("::");
    s.push_str(item.name.get_ref().as_slice());
    return s;
}

fn blank<'a>(s: Option<&'a str>) -> &'a str {
    match s {
        Some(s) => s,
        None => ""
    }
}

fn shorter<'a>(s: Option<&'a str>) -> &'a str {
    match s {
        Some(s) => match s.find_str("\n\n") {
            Some(pos) => s.slice_to(pos),
            None => s,
        },
        None => ""
    }
}

fn document(w: &mut io::Writer, item: &clean::Item) {
    match item.doc_value() {
        Some(s) => {
            write!(w, "<div class='docblock'>{}</div>", Markdown(s));
        }
        None => {}
    }
}

fn item_module(w: &mut io::Writer, cx: &Context,
               item: &clean::Item, items: &[clean::Item]) {
    document(w, item);
    debug2!("{:?}", items);
    let mut indices = vec::from_fn(items.len(), |i| i);

    fn lt(i1: &clean::Item, i2: &clean::Item, idx1: uint, idx2: uint) -> bool {
        if shortty(i1) == shortty(i2) {
            return i1.name < i2.name;
        }
        match (&i1.inner, &i2.inner) {
            (&clean::ViewItemItem(ref a), &clean::ViewItemItem(ref b)) => {
                match (&a.inner, &b.inner) {
                    (&clean::ExternMod(*), _) => true,
                    (_, &clean::ExternMod(*)) => false,
                    _ => idx1 < idx2,
                }
            }
            (&clean::ViewItemItem(*), _) => true,
            (_, &clean::ViewItemItem(*)) => false,
            (&clean::ModuleItem(*), _) => true,
            (_, &clean::ModuleItem(*)) => false,
            (&clean::StructItem(*), _) => true,
            (_, &clean::StructItem(*)) => false,
            (&clean::EnumItem(*), _) => true,
            (_, &clean::EnumItem(*)) => false,
            (&clean::StaticItem(*), _) => true,
            (_, &clean::StaticItem(*)) => false,
            (&clean::ForeignFunctionItem(*), _) => true,
            (_, &clean::ForeignFunctionItem(*)) => false,
            (&clean::ForeignStaticItem(*), _) => true,
            (_, &clean::ForeignStaticItem(*)) => false,
            (&clean::TraitItem(*), _) => true,
            (_, &clean::TraitItem(*)) => false,
            (&clean::FunctionItem(*), _) => true,
            (_, &clean::FunctionItem(*)) => false,
            (&clean::TypedefItem(*), _) => true,
            (_, &clean::TypedefItem(*)) => false,
            _ => idx1 < idx2,
        }
    }

    debug2!("{:?}", indices);
    do sort::quick_sort(indices) |&i1, &i2| {
        lt(&items[i1], &items[i2], i1, i2)
    }

    debug2!("{:?}", indices);
    let mut curty = "";
    for &idx in indices.iter() {
        let myitem = &items[idx];

        let myty = shortty(myitem);
        if myty != curty {
            if curty != "" {
                write!(w, "</table>");
            }
            curty = myty;
            write!(w, "<h2>{}</h2>\n<table>", match myitem.inner {
                clean::ModuleItem(*)          => "Modules",
                clean::StructItem(*)          => "Structs",
                clean::EnumItem(*)            => "Enums",
                clean::FunctionItem(*)        => "Functions",
                clean::TypedefItem(*)         => "Type Definitions",
                clean::StaticItem(*)          => "Statics",
                clean::TraitItem(*)           => "Traits",
                clean::ImplItem(*)            => "Implementations",
                clean::ViewItemItem(*)        => "Reexports",
                clean::TyMethodItem(*)        => "Type Methods",
                clean::MethodItem(*)          => "Methods",
                clean::StructFieldItem(*)     => "Struct Fields",
                clean::VariantItem(*)         => "Variants",
                clean::ForeignFunctionItem(*) => "Foreign Functions",
                clean::ForeignStaticItem(*)   => "Foreign Statics",
            });
        }

        match myitem.inner {
            clean::StaticItem(ref s) | clean::ForeignStaticItem(ref s) => {
                struct Initializer<'self>(&'self str);
                impl<'self> fmt::Default for Initializer<'self> {
                    fn fmt(s: &Initializer<'self>, f: &mut fmt::Formatter) {
                        if s.len() == 0 { return; }
                        write!(f.buf, "<code> = </code>");
                        let tag = if s.contains("\n") { "pre" } else { "code" };
                        write!(f.buf, "<{tag}>{}</{tag}>",
                               s.as_slice(), tag=tag);
                    }
                }

                write!(w, "
                    <tr>
                        <td><code>{}static {}: {}</code>{}</td>
                        <td class='docblock'>{}&nbsp;</td>
                    </tr>
                ",
                VisSpace(myitem.visibility),
                *myitem.name.get_ref(),
                s.type_,
                Initializer(s.expr),
                Markdown(blank(myitem.doc_value())));
            }

            clean::ViewItemItem(ref item) => {
                match item.inner {
                    clean::ExternMod(ref name, ref src, _, _) => {
                        write!(w, "<tr><td><code>extern mod {}",
                               name.as_slice());
                        match *src {
                            Some(ref src) => write!(w, " = \"{}\"",
                                                    src.as_slice()),
                            None => {}
                        }
                        write!(w, ";</code></td></tr>");
                    }

                    clean::Import(ref imports) => {
                        for import in imports.iter() {
                            write!(w, "<tr><td><code>{}{}</code></td></tr>",
                                   VisSpace(myitem.visibility),
                                   *import);
                        }
                    }
                }

            }

            _ => {
                if myitem.name.is_none() { continue }
                write!(w, "
                    <tr>
                        <td><a class='{class}' href='{href}'
                               title='{title}'>{}</a></td>
                        <td class='docblock short'>{}</td>
                    </tr>
                ",
                *myitem.name.get_ref(),
                Markdown(shorter(myitem.doc_value())),
                class = shortty(myitem),
                href = item_path(myitem),
                title = full_path(cx, myitem));
            }
        }
    }
    write!(w, "</table>");
}

fn item_function(w: &mut io::Writer, it: &clean::Item, f: &clean::Function) {
    write!(w, "<pre class='fn'>{vis}{purity}fn {name}{generics}{decl}</pre>",
           vis = VisSpace(it.visibility),
           purity = PuritySpace(f.purity),
           name = it.name.get_ref().as_slice(),
           generics = f.generics,
           decl = f.decl);
    document(w, it);
}

fn item_trait(w: &mut io::Writer, it: &clean::Item, t: &clean::Trait) {
    let mut parents = ~"";
    if t.parents.len() > 0 {
        parents.push_str(": ");
        for (i, p) in t.parents.iter().enumerate() {
            if i > 0 { parents.push_str(" + "); }
            parents.push_str(format!("{}", *p));
        }
    }

    // Output the trait definition
    write!(w, "<pre class='trait'>{}trait {}{}{} ",
           VisSpace(it.visibility),
           it.name.get_ref().as_slice(),
           t.generics,
           parents);
    let required = t.methods.iter().filter(|m| m.is_req()).to_owned_vec();
    let provided = t.methods.iter().filter(|m| !m.is_req()).to_owned_vec();

    if t.methods.len() == 0 {
        write!(w, "\\{ \\}");
    } else {
        write!(w, "\\{\n");
        for m in required.iter() {
            write!(w, "    ");
            render_method(w, m.item(), true);
            write!(w, ";\n");
        }
        if required.len() > 0 && provided.len() > 0 {
            w.write("\n".as_bytes());
        }
        for m in provided.iter() {
            write!(w, "    ");
            render_method(w, m.item(), true);
            write!(w, " \\{ ... \\}\n");
        }
        write!(w, "\\}");
    }
    write!(w, "</pre>");

    // Trait documentation
    document(w, it);

    fn meth(w: &mut io::Writer, m: &clean::TraitMethod) {
        write!(w, "<h3 id='{}.{}' class='method'><code>",
               shortty(m.item()),
               *m.item().name.get_ref());
        render_method(w, m.item(), false);
        write!(w, "</code></h3>");
        document(w, m.item());
    }

    // Output the documentation for each function individually
    if required.len() > 0 {
        write!(w, "
            <h2 id='required-methods'>Required Methods</h2>
            <div class='methods'>
        ");
        for m in required.iter() {
            meth(w, *m);
        }
        write!(w, "</div>");
    }
    if provided.len() > 0 {
        write!(w, "
            <h2 id='provided-methods'>Provided Methods</h2>
            <div class='methods'>
        ");
        for m in provided.iter() {
            meth(w, *m);
        }
        write!(w, "</div>");
    }

    do local_data::get(cache_key) |cache| {
        do cache.unwrap().read |cache| {
            match cache.implementors.find(&it.id) {
                Some(implementors) => {
                    write!(w, "
                        <h2 id='implementors'>Implementors</h2>
                        <ul class='item-list'>
                    ");
                    for i in implementors.iter() {
                        match *i {
                            PathType(ref ty) => {
                                write!(w, "<li><code>{}</code></li>", *ty);
                            }
                            OtherType(ref generics, ref trait_, ref for_) => {
                                write!(w, "<li><code>impl{} {} for {}</code></li>",
                                       *generics, *trait_, *for_);
                            }
                        }
                    }
                    write!(w, "</ul>");
                }
                None => {}
            }
        }
    }
}

fn render_method(w: &mut io::Writer, meth: &clean::Item, withlink: bool) {
    fn fun(w: &mut io::Writer, it: &clean::Item, purity: ast::purity,
           g: &clean::Generics, selfty: &clean::SelfTy, d: &clean::FnDecl,
           withlink: bool) {
        write!(w, "{}fn {withlink, select,
                            true{<a href='\\#{ty}.{name}'
                                    class='fnname'>{name}</a>}
                            other{<span class='fnname'>{name}</span>}
                        }{generics}{decl}",
               match purity {
                   ast::unsafe_fn => "unsafe ",
                   _ => "",
               },
               ty = shortty(it),
               name = it.name.get_ref().as_slice(),
               generics = *g,
               decl = Method(selfty, d),
               withlink = if withlink {"true"} else {"false"});
    }
    match meth.inner {
        clean::TyMethodItem(ref m) => {
            fun(w, meth, m.purity, &m.generics, &m.self_, &m.decl, withlink);
        }
        clean::MethodItem(ref m) => {
            fun(w, meth, m.purity, &m.generics, &m.self_, &m.decl, withlink);
        }
        _ => unreachable!()
    }
}

fn item_struct(w: &mut io::Writer, it: &clean::Item, s: &clean::Struct) {
    write!(w, "<pre class='struct'>");
    render_struct(w, it, Some(&s.generics), s.struct_type, s.fields, "", true);
    write!(w, "</pre>");

    document(w, it);
    match s.struct_type {
        doctree::Plain => {
            write!(w, "<h2 class='fields'>Fields</h2>\n<table>");
            for field in s.fields.iter() {
                write!(w, "<tr><td id='structfield.{name}'>\
                                <code>{name}</code></td><td>",
                       name = field.name.get_ref().as_slice());
                document(w, field);
                write!(w, "</td></tr>");
            }
            write!(w, "</table>");
        }
        _ => {}
    }
    render_methods(w, it);
}

fn item_enum(w: &mut io::Writer, it: &clean::Item, e: &clean::Enum) {
    write!(w, "<pre class='enum'>{}enum {}{}",
           VisSpace(it.visibility),
           it.name.get_ref().as_slice(),
           e.generics);
    if e.variants.len() == 0 {
        write!(w, " \\{\\}");
    } else {
        write!(w, " \\{\n");
        for v in e.variants.iter() {
            write!(w, "    ");
            let name = v.name.get_ref().as_slice();
            match v.inner {
                clean::VariantItem(ref var) => {
                    match var.kind {
                        clean::CLikeVariant => write!(w, "{}", name),
                        clean::TupleVariant(ref tys) => {
                            write!(w, "{}(", name);
                            for (i, ty) in tys.iter().enumerate() {
                                if i > 0 { write!(w, ", ") }
                                write!(w, "{}", *ty);
                            }
                            write!(w, ")");
                        }
                        clean::StructVariant(ref s) => {
                            render_struct(w, v, None, s.struct_type, s.fields,
                                          "    ", false);
                        }
                    }
                }
                _ => unreachable!()
            }
            write!(w, ",\n");
        }
        write!(w, "\\}");
    }
    write!(w, "</pre>");

    document(w, it);
    if e.variants.len() > 0 {
        write!(w, "<h2 class='variants'>Variants</h2>\n<table>");
        for variant in e.variants.iter() {
            write!(w, "<tr><td id='variant.{name}'><code>{name}</code></td><td>",
                   name = variant.name.get_ref().as_slice());
            document(w, variant);
            write!(w, "</td></tr>");
        }
        write!(w, "</table>");

    }
    render_methods(w, it);
}

fn render_struct(w: &mut io::Writer, it: &clean::Item,
                 g: Option<&clean::Generics>,
                 ty: doctree::StructType,
                 fields: &[clean::Item],
                 tab: &str,
                 structhead: bool) {
    write!(w, "{}{}{}",
           VisSpace(it.visibility),
           if structhead {"struct "} else {""},
           it.name.get_ref().as_slice());
    match g {
        Some(g) => write!(w, "{}", *g),
        None => {}
    }
    match ty {
        doctree::Plain => {
            write!(w, " \\{\n{}", tab);
            for field in fields.iter() {
                match field.inner {
                    clean::StructFieldItem(ref ty) => {
                        write!(w, "    {}{}: {},\n{}",
                               VisSpace(field.visibility),
                               field.name.get_ref().as_slice(),
                               ty.type_,
                               tab);
                    }
                    _ => unreachable!()
                }
            }
            write!(w, "\\}");
        }
        doctree::Tuple | doctree::Newtype => {
            write!(w, "(");
            for (i, field) in fields.iter().enumerate() {
                if i > 0 { write!(w, ", ") }
                match field.inner {
                    clean::StructFieldItem(ref field) => {
                        write!(w, "{}", field.type_);
                    }
                    _ => unreachable!()
                }
            }
            write!(w, ");");
        }
        doctree::Unit => { write!(w, ";"); }
    }
}

fn render_methods(w: &mut io::Writer, it: &clean::Item) {
    do local_data::get(cache_key) |cache| {
        let cache = cache.unwrap();
        do cache.read |c| {
            match c.impls.find(&it.id) {
                Some(v) => {
                    let mut non_trait = v.iter().filter(|p| {
                        p.n0_ref().trait_.is_none()
                    });
                    let non_trait = non_trait.to_owned_vec();
                    let mut traits = v.iter().filter(|p| {
                        p.n0_ref().trait_.is_some()
                    });
                    let traits = traits.to_owned_vec();

                    if non_trait.len() > 0 {
                        write!(w, "<h2 id='methods'>Methods</h2>");
                        for &(ref i, ref dox) in non_trait.move_iter() {
                            render_impl(w, i, dox);
                        }
                    }
                    if traits.len() > 0 {
                        write!(w, "<h2 id='implementations'>Trait \
                                   Implementations</h2>");
                        for &(ref i, ref dox) in traits.move_iter() {
                            render_impl(w, i, dox);
                        }
                    }
                }
                None => {}
            }
        }
    }
}

fn render_impl(w: &mut io::Writer, i: &clean::Impl, dox: &Option<~str>) {
    write!(w, "<h3 class='impl'><code>impl{} ", i.generics);
    let trait_id = match i.trait_ {
        Some(ref ty) => {
            write!(w, "{} for ", *ty);
            match *ty {
                clean::ResolvedPath { id, _ } => Some(id),
                _ => None,
            }
        }
        None => None
    };
    write!(w, "{}</code></h3>", i.for_);
    match *dox {
        Some(ref dox) => {
            write!(w, "<div class='docblock'>{}</div>",
                   Markdown(dox.as_slice()));
        }
        None => {}
    }
    write!(w, "<div class='methods'>");
    for meth in i.methods.iter() {
        write!(w, "<h4 id='method.{}' class='method'><code>",
               *meth.name.get_ref());
        render_method(w, meth, false);
        write!(w, "</code></h4>\n");
        match meth.doc_value() {
            Some(s) => {
                write!(w, "<div class='docblock'>{}</div>", Markdown(s));
                continue
            }
            None => {}
        }

        // No documentation? Attempt to slurp in the trait's documentation
        let trait_id = match trait_id {
            None => continue,
            Some(id) => id,
        };
        do local_data::get(cache_key) |cache| {
            do cache.unwrap().read |cache| {
                let name = meth.name.get_ref().as_slice();
                match cache.traits.find(&trait_id) {
                    Some(m) => {
                        match m.find_equiv(&name) {
                            Some(s) => {
                                write!(w, "<div class='docblock'>{}</div>",
                                       Markdown(s.as_slice()));
                            }
                            None => {}
                        }
                    }
                    None => {}
                }
            }
        }
    }
    write!(w, "</div>");
}

fn item_typedef(w: &mut io::Writer, it: &clean::Item, t: &clean::Typedef) {
    write!(w, "<pre class='typedef'>type {}{} = {};</pre>",
           it.name.get_ref().as_slice(),
           t.generics,
           t.type_);

    document(w, it);
}

impl<'self> fmt::Default for Sidebar<'self> {
    fn fmt(s: &Sidebar<'self>, fmt: &mut fmt::Formatter) {
        let cx = s.cx;
        let it = s.item;
        write!(fmt.buf, "<p class='location'>");
        let len = cx.current.len() - if it.is_mod() {1} else {0};
        for (i, name) in cx.current.iter().take(len).enumerate() {
            if i > 0 { write!(fmt.buf, "&\\#8203;::") }
            write!(fmt.buf, "<a href='{}index.html'>{}</a>",
                   cx.root_path.slice_to((cx.current.len() - i - 1) * 3), *name);
        }
        write!(fmt.buf, "</p>");

        fn block(w: &mut io::Writer, short: &str, longty: &str,
                 cur: &clean::Item, cx: &Context) {
            let items = match cx.sidebar.find_equiv(&short) {
                Some(items) => items.as_slice(),
                None => return
            };
            write!(w, "<div class='block {}'><h2>{}</h2>", short, longty);
            for item in items.iter() {
                let class = if cur.name.get_ref() == item &&
                               short == shortty(cur) { "current" } else { "" };
                write!(w, "<a class='{ty} {class}' href='{curty, select,
                                mod{../}
                                other{}
                           }{tysel, select,
                                mod{{name}/index.html}
                                other{#.{name}.html}
                           }'>{name}</a><br/>",
                       ty = short,
                       tysel = short,
                       class = class,
                       curty = shortty(cur),
                       name = item.as_slice());
            }
            write!(w, "</div>");
        }

        block(fmt.buf, "mod", "Modules", it, cx);
        block(fmt.buf, "struct", "Structs", it, cx);
        block(fmt.buf, "enum", "Enums", it, cx);
        block(fmt.buf, "trait", "Traits", it, cx);
        block(fmt.buf, "fn", "Functions", it, cx);
    }
}

fn build_sidebar(m: &clean::Module) -> HashMap<~str, ~[~str]> {
    let mut map = HashMap::new();
    for item in m.items.iter() {
        let short = shortty(item);
        let myname = match item.name {
            None => continue,
            Some(ref s) => s.to_owned(),
        };
        let v = map.find_or_insert_with(short.to_owned(), |_| ~[]);
        v.push(myname);
    }

    for (_, items) in map.mut_iter() {
        sort::quick_sort(*items, |i1, i2| i1 < i2);
    }
    return map;
}

impl<'self> fmt::Default for Source<'self> {
    fn fmt(s: &Source<'self>, fmt: &mut fmt::Formatter) {
        let lines = s.line_iter().len();
        let mut cols = 0;
        let mut tmp = lines;
        while tmp > 0 {
            cols += 1;
            tmp /= 10;
        }
        write!(fmt.buf, "<pre class='line-numbers'>");
        for i in range(1, lines + 1) {
            write!(fmt.buf, "<span id='{0:u}'>{0:1$u}</span>\n", i, cols);
        }
        write!(fmt.buf, "</pre>");
        write!(fmt.buf, "<pre class='rust'>");
        write!(fmt.buf, "{}", Escape(s.as_slice()));
        write!(fmt.buf, "</pre>");
    }
}
