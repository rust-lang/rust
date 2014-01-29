// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rustdoc's HTML Rendering module
//!
//! This modules contains the bulk of the logic necessary for rendering a
//! rustdoc `clean::Crate` instance to a set of static HTML pages. This
//! rendering process is largely driven by the `format!` syntax extension to
//! perform all I/O into files and streams.
//!
//! The rendering process is largely driven by the `Context` and `Cache`
//! structures. The cache is pre-populated by crawling the crate in question,
//! and then it is shared among the various rendering tasks. The cache is meant
//! to be a fairly large structure not implementing `Clone` (because it's shared
//! among tasks). The context, however, should be a lightweight structure. This
//! is cloned per-task and contains information about what is currently being
//! rendered.
//!
//! In order to speed up rendering (mostly because of markdown rendering), the
//! rendering process has been parallelized. This parallelization is only
//! exposed through the `crate` method on the context, and then also from the
//! fact that the shared cache is stored in TLS (and must be accessed as such).
//!
//! In addition to rendering the crate itself, this module is also responsible
//! for creating the corresponding search index and source file renderings.
//! These tasks are not parallelized (they haven't been a bottleneck yet), and
//! both occur before the crate is rendered.

use std::fmt;
use std::hashmap::{HashMap, HashSet};
use std::local_data;
use std::io;
use std::io::{fs, File, BufferedWriter};
use std::str;
use std::vec;

use extra::arc::Arc;
use extra::json::ToJson;
use syntax::ast;
use syntax::attr;
use syntax::parse::token::InternedString;

use clean;
use doctree;
use fold::DocFolder;
use html::escape::Escape;
use html::format::{VisSpace, Method, PuritySpace};
use html::layout;
use html::markdown::Markdown;

/// Major driving force in all rustdoc rendering. This contains information
/// about where in the tree-like hierarchy rendering is occurring and controls
/// how the current page is being rendered.
///
/// It is intended that this context is a lightweight object which can be fairly
/// easily cloned because it is cloned per work-job (about once per item in the
/// rustdoc tree).
#[deriving(Clone)]
pub struct Context {
    /// Current hierarchy of components leading down to what's currently being
    /// rendered
    current: ~[~str],
    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    root_path: ~str,
    /// The current destination folder of where HTML artifacts should be placed.
    /// This changes as the context descends into the module hierarchy.
    dst: Path,
    /// This describes the layout of each page, and is not modified after
    /// creation of the context (contains info like the favicon)
    layout: layout::Layout,
    /// This map is a list of what should be displayed on the sidebar of the
    /// current page. The key is the section header (traits, modules,
    /// functions), and the value is the list of containers belonging to this
    /// header. This map will change depending on the surrounding context of the
    /// page.
    sidebar: HashMap<~str, ~[~str]>,
    /// This flag indicates whether [src] links should be generated or not. If
    /// the source files are present in the html rendering, then this will be
    /// `true`.
    include_sources: bool,
}

/// Indicates where an external crate can be found.
pub enum ExternalLocation {
    /// Remote URL root of the external crate
    Remote(~str),
    /// This external crate can be found in the local doc/ folder
    Local,
    /// The external crate could not be found.
    Unknown,
}

/// Different ways an implementor of a trait can be rendered.
enum Implementor {
    /// Paths are displayed specially by omitting the `impl XX for` cruft
    PathType(clean::Type),
    /// This is the generic representation of a trait implementor, used for
    /// primitive types and otherwise non-path types.
    OtherType(clean::Generics, /* trait */ clean::Type, /* for */ clean::Type),
}

/// This cache is used to store information about the `clean::Crate` being
/// rendered in order to provide more useful documentation. This contains
/// information like all implementors of a trait, all traits a type implements,
/// documentation for all known traits, etc.
///
/// This structure purposefully does not implement `Clone` because it's intended
/// to be a fairly large and expensive structure to clone. Instead this adheres
/// to both `Send` and `Freeze` so it may be stored in a `Arc` instance and
/// shared among the various rendering tasks.
pub struct Cache {
    /// Mapping of typaram ids to the name of the type parameter. This is used
    /// when pretty-printing a type (so pretty printing doesn't have to
    /// painfully maintain a context like this)
    typarams: HashMap<ast::NodeId, ~str>,

    /// Maps a type id to all known implementations for that type. This is only
    /// recognized for intra-crate `ResolvedPath` types, and is used to print
    /// out extra documentation on the page of an enum/struct.
    ///
    /// The values of the map are a list of implementations and documentation
    /// found on that implementation.
    impls: HashMap<ast::NodeId, ~[(clean::Impl, Option<~str>)]>,

    /// Maintains a mapping of local crate node ids to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    paths: HashMap<ast::NodeId, (~[~str], &'static str)>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    traits: HashMap<ast::NodeId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    implementors: HashMap<ast::NodeId, ~[Implementor]>,

    /// Cache of where external crate documentation can be found.
    extern_locations: HashMap<ast::CrateNum, ExternalLocation>,

    // Private fields only used when initially crawling a crate to build a cache

    priv stack: ~[~str],
    priv parent_stack: ~[ast::NodeId],
    priv search_index: ~[IndexItem],
    priv privmod: bool,
}

/// Helper struct to render all source code to HTML pages
struct SourceCollector<'a> {
    cx: &'a mut Context,

    /// Processed source-file paths
    seen: HashSet<~str>,
    /// Root destination to place all HTML output into
    dst: Path,
}

/// Wrapper struct to render the source code of a file. This will do things like
/// adding line numbers to the left-hand side.
struct Source<'a>(&'a str);

// Helper structs for rendering items/sidebars and carrying along contextual
// information

struct Item<'a> { cx: &'a Context, item: &'a clean::Item, }
struct Sidebar<'a> { cx: &'a Context, item: &'a clean::Item, }

/// Struct representing one entry in the JS search index. These are all emitted
/// by hand to a large JS file at the end of cache-creation.
struct IndexItem {
    ty: &'static str,
    name: ~str,
    path: ~str,
    desc: ~str,
    parent: Option<ast::NodeId>,
}

// TLS keys used to carry information around during rendering.

local_data_key!(pub cache_key: Arc<Cache>)
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

    match crate.module.as_ref().map(|m| m.doc_list().unwrap_or(&[])) {
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
        privmod: false,
    };
    cache.stack.push(crate.name.clone());
    crate = cache.fold_crate(crate);

    // Add all the static files
    let mut dst = cx.dst.join(crate.name.as_slice());
    mkdir(&dst);
    write(dst.join("jquery.js"), include_str!("static/jquery-2.0.3.min.js"));
    write(dst.join("main.js"), include_str!("static/main.js"));
    write(dst.join("main.css"), include_str!("static/main.css"));
    write(dst.join("normalize.css"), include_str!("static/normalize.css"));

    // Publish the search index
    {
        dst.push("search-index.js");
        let mut w = BufferedWriter::new(File::create(&dst).unwrap());
        let w = &mut w as &mut Writer;
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
            write!(w, "'{}':\\{type:'{}',name:'{}'\\}",
                   id, short, *fqp.last().unwrap());
        }
        write!(w, "\\};");
        w.flush();
    }

    // Render all source files (this may turn into a giant no-op)
    {
        info!("emitting source files");
        let dst = cx.dst.join("src");
        mkdir(&dst);
        let dst = dst.join(crate.name.as_slice());
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

/// Writes the entire contents of a string to a destination, not attempting to
/// catch any errors.
fn write(dst: Path, contents: &str) {
    File::create(&dst).write(contents.as_bytes());
}

/// Makes a directory on the filesystem, failing the task if an error occurs and
/// skipping if the directory already exists.
fn mkdir(path: &Path) {
    io::io_error::cond.trap(|err| {
        error!("Couldn't create directory `{}`: {}",
                path.display(), err.desc);
        fail!()
    }).inside(|| {
        if !path.is_dir() {
            fs::mkdir(path, io::UserRWX);
        }
    })
}

/// Takes a path to a source file and cleans the path to it. This canonicalizes
/// things like ".." to components which preserve the "top down" hierarchy of a
/// static HTML tree.
// FIXME (#9639): The closure should deal with &[u8] instead of &str
fn clean_srcpath(src: &[u8], f: |&str|) {
    let p = Path::new(src);
    if p.as_vec() != bytes!(".") {
        for c in p.str_components().map(|x|x.unwrap()) {
            if ".." == c {
                f("up");
            } else {
                f(c.as_slice())
            }
        }
    }
}

/// Attempts to find where an external crate is located, given that we're
/// rendering in to the specified source destination.
fn extern_location(e: &clean::ExternalCrate, dst: &Path) -> ExternalLocation {
    // See if there's documentation generated into the local directory
    let local_location = dst.join(e.name.as_slice());
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

impl<'a> DocFolder for SourceCollector<'a> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        // If we're including source files, and we haven't seen this file yet,
        // then we need to render it out to the filesystem
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

impl<'a> SourceCollector<'a> {
    /// Renders the given filename into its corresponding HTML source file.
    fn emit_source(&mut self, filename: &str) -> bool {
        let p = Path::new(filename);

        // Read the contents of the file
        let mut contents = ~[];
        {
            let mut buf = [0, ..1024];
            // If we couldn't open this file, then just returns because it
            // probably means that it's some standard library macro thing and we
            // can't have the source to it anyway.
            let mut r = match io::result(|| File::open(&p)) {
                Ok(r) => r,
                // eew macro hacks
                Err(..) => return filename == "<std-macros>"
            };

            // read everything
            loop {
                match r.read(buf) {
                    Some(n) => contents.push_all(buf.slice_to(n)),
                    None => break
                }
            }
        }
        let contents = str::from_utf8_owned(contents).unwrap();

        // Create the intermediate directories
        let mut cur = self.dst.clone();
        let mut root_path = ~"../../";
        clean_srcpath(p.dirname(), |component| {
            cur.push(component);
            mkdir(&cur);
            root_path.push_str("../");
        });

        cur.push(p.filename().expect("source has no filename") + bytes!(".html"));
        let mut w = BufferedWriter::new(File::create(&cur).unwrap());

        let title = cur.filename_display().with_str(|s| format!("{} -- source", s));
        let page = layout::Page {
            title: title,
            ty: "source",
            root_path: root_path,
        };
        layout::render(&mut w as &mut Writer, &self.cx.layout,
                       &page, &(""), &Source(contents.as_slice()));
        w.flush();
        return true;
    }
}

impl DocFolder for Cache {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        // If this is a private module, we don't want it in the search index.
        let orig_privmod = match item.inner {
            clean::ModuleItem(..) => {
                let prev = self.privmod;
                self.privmod = prev || item.visibility != Some(ast::Public);
                prev
            }
            _ => self.privmod,
        };

        // Register any generics to their corresponding string. This is used
        // when pretty-printing types
        match item.inner {
            clean::StructItem(ref s)          => self.generics(&s.generics),
            clean::EnumItem(ref e)            => self.generics(&e.generics),
            clean::FunctionItem(ref f)        => self.generics(&f.generics),
            clean::TypedefItem(ref t)         => self.generics(&t.generics),
            clean::TraitItem(ref t)           => self.generics(&t.generics),
            clean::ImplItem(ref i)            => self.generics(&i.generics),
            clean::TyMethodItem(ref i)        => self.generics(&i.generics),
            clean::MethodItem(ref i)          => self.generics(&i.generics),
            clean::ForeignFunctionItem(ref f) => self.generics(&f.generics),
            _ => {}
        }

        // Propagate a trait methods' documentation to all implementors of the
        // trait
        match item.inner {
            clean::TraitItem(ref t) => {
                self.traits.insert(item.id, t.clone());
            }
            _ => {}
        }

        // Collect all the implementors of traits.
        match item.inner {
            clean::ImplItem(ref i) => {
                match i.trait_ {
                    Some(clean::ResolvedPath{ id, .. }) => {
                        let v = self.implementors.find_or_insert_with(id, |_|{
                            ~[]
                        });
                        match i.for_ {
                            clean::ResolvedPath{..} => {
                                v.unshift(PathType(i.for_.clone()));
                            }
                            _ => {
                                v.push(OtherType(i.generics.clone(),
                                                 i.trait_.get_ref().clone(),
                                                 i.for_.clone()));
                            }
                        }
                    }
                    Some(..) | None => {}
                }
            }
            _ => {}
        }

        // Index this method for searching later on
        match item.name {
            Some(ref s) => {
                let parent = match item.inner {
                    clean::TyMethodItem(..) |
                    clean::StructFieldItem(..) |
                    clean::VariantItem(..) => {
                        Some((Some(*self.parent_stack.last().unwrap()),
                              self.stack.slice_to(self.stack.len() - 1)))

                    }
                    clean::MethodItem(..) => {
                        if self.parent_stack.len() == 0 {
                            None
                        } else {
                            let last = self.parent_stack.last().unwrap();
                            let amt = match self.paths.find(last) {
                                Some(&(_, "trait")) => self.stack.len() - 1,
                                Some(..) | None => self.stack.len(),
                            };
                            Some((Some(*last), self.stack.slice_to(amt)))
                        }
                    }
                    _ => Some((None, self.stack.as_slice()))
                };
                match parent {
                    Some((parent, path)) if !self.privmod => {
                        self.search_index.push(IndexItem {
                            ty: shortty(&item),
                            name: s.to_owned(),
                            path: path.connect("::"),
                            desc: shorter(item.doc_value()).to_owned(),
                            parent: parent,
                        });
                    }
                    Some(..) | None => {}
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
            clean::StructItem(..) | clean::EnumItem(..) |
            clean::TypedefItem(..) | clean::TraitItem(..) |
            clean::FunctionItem(..) | clean::ModuleItem(..) |
            clean::ForeignFunctionItem(..) | clean::VariantItem(..) => {
                self.paths.insert(item.id, (self.stack.clone(), shortty(&item)));
            }
            _ => {}
        }

        // Maintain the parent stack
        let parent_pushed = match item.inner {
            clean::TraitItem(..) | clean::EnumItem(..) | clean::StructItem(..) => {
                self.parent_stack.push(item.id); true
            }
            clean::ImplItem(ref i) => {
                match i.for_ {
                    clean::ResolvedPath{ id, .. } => {
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
                    clean::Item{ attrs, inner: clean::ImplItem(i), .. } => {
                        match i.for_ {
                            clean::ResolvedPath { id, .. } => {
                                let v = self.impls.find_or_insert_with(id, |_| {
                                    ~[]
                                });
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
                                    Some(..) | None => {
                                        v.push((i, None));
                                    }
                                }
                            }
                            _ => {}
                        }
                        None
                    }
                    // Private modules may survive the strip-private pass if
                    // they contain impls for public types, but those will get
                    // stripped here
                    clean::Item { inner: clean::ModuleItem(ref m),
                                  visibility, .. }
                            if (m.items.len() == 0 &&
                                item.doc_value().is_none()) ||
                               visibility != Some(ast::Public) => None,

                    i => Some(i),
                }
            }
            i => i,
        };

        if pushed { self.stack.pop().unwrap(); }
        if parent_pushed { self.parent_stack.pop().unwrap(); }
        self.privmod = orig_privmod;
        return ret;
    }
}

impl<'a> Cache {
    fn generics(&mut self, generics: &clean::Generics) {
        for typ in generics.type_params.iter() {
            self.typarams.insert(typ.id, typ.name.clone());
        }
    }
}

impl Context {
    /// Recurse in the directory structure and change the "root path" to make
    /// sure it always points to the top (relatively)
    fn recurse<T>(&mut self, s: ~str, f: |&mut Context| -> T) -> T {
        if s.len() == 0 {
            fail!("what {:?}", self);
        }
        let prev = self.dst.clone();
        self.dst.push(s.as_slice());
        self.root_path.push_str("../");
        self.current.push(s);

        info!("Recursing into {}", self.dst.display());

        mkdir(&self.dst);
        let ret = f(self);

        info!("Recursed; leaving {}", self.dst.display());

        // Go back to where we were at
        self.dst = prev;
        let len = self.root_path.len();
        self.root_path.truncate(len - 3);
        self.current.pop().unwrap();

        return ret;
    }

    /// Main method for rendering a crate.
    ///
    /// This currently isn't parallelized, but it'd be pretty easy to add
    /// parallelization to this function.
    fn crate(self, mut crate: clean::Crate, cache: Cache) {
        let mut item = match crate.module.take() {
            Some(i) => i,
            None => return
        };
        item.name = Some(crate.name);

        // using a rwarc makes this parallelizable in the future
        local_data::set(cache_key, Arc::new(cache));

        let mut work = ~[(self, item)];
        loop {
            match work.pop() {
                Some((mut cx, item)) => cx.item(item, |cx, item| {
                    work.push((cx.clone(), item));
                }),
                None => break,
            }
        }
    }

    /// Non-parellelized version of rendering an item. This will take the input
    /// item, render its contents, and then invoke the specified closure with
    /// all sub-items which need to be rendered.
    ///
    /// The rendering driver uses this closure to queue up more work.
    fn item(&mut self, item: clean::Item, f: |&mut Context, clean::Item|) {
        fn render(w: io::File, cx: &mut Context, it: &clean::Item,
                  pushname: bool) {
            info!("Rendering an item to {}", w.path().display());
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
            layout::render(&mut writer as &mut Writer, &cx.layout, &page,
                           &Sidebar{ cx: cx, item: it },
                           &Item{ cx: cx, item: it });
            writer.flush();
        }

        match item.inner {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            clean::ModuleItem(..) => {
                let name = item.name.get_ref().to_owned();
                let mut item = Some(item);
                self.recurse(name, |this| {
                    let item = item.take_unwrap();
                    let dst = this.dst.join("index.html");
                    render(File::create(&dst).unwrap(), this, &item, false);

                    let m = match item.inner {
                        clean::ModuleItem(m) => m,
                        _ => unreachable!()
                    };
                    this.sidebar = build_sidebar(&m);
                    for item in m.items.move_iter() {
                        f(this,item);
                    }
                })
            }

            // Things which don't have names (like impls) don't get special
            // pages dedicated to them.
            _ if item.name.is_some() => {
                let dst = self.dst.join(item_path(&item));
                render(File::create(&dst).unwrap(), self, &item, true);
            }

            _ => {}
        }
    }
}

fn shortty(item: &clean::Item) -> &'static str {
    match item.inner {
        clean::ModuleItem(..)          => "mod",
        clean::StructItem(..)          => "struct",
        clean::EnumItem(..)            => "enum",
        clean::FunctionItem(..)        => "fn",
        clean::TypedefItem(..)         => "typedef",
        clean::StaticItem(..)          => "static",
        clean::TraitItem(..)           => "trait",
        clean::ImplItem(..)            => "impl",
        clean::ViewItemItem(..)        => "viewitem",
        clean::TyMethodItem(..)        => "tymethod",
        clean::MethodItem(..)          => "method",
        clean::StructFieldItem(..)     => "structfield",
        clean::VariantItem(..)         => "variant",
        clean::ForeignFunctionItem(..) => "ffi",
        clean::ForeignStaticItem(..)   => "ffs",
    }
}

impl<'a> Item<'a> {
    fn ismodule(&self) -> bool {
        match self.item.inner {
            clean::ModuleItem(..) => true, _ => false
        }
    }
}

impl<'a> fmt::Show for Item<'a> {
    fn fmt(it: &Item<'a>, fmt: &mut fmt::Formatter) {
        match attr::find_stability(it.item.attrs.iter()) {
            Some(ref stability) => {
                write!(fmt.buf,
                       "<a class='stability {lvl}' title='{reason}'>{lvl}</a>",
                       lvl = stability.level.to_str(),
                       reason = match stability.text {
                           Some(ref s) => (*s).clone(),
                           None => InternedString::new(""),
                       });
            }
            None => {}
        }

        if it.cx.include_sources {
            let mut path = ~[];
            clean_srcpath(it.item.source.filename.as_bytes(), |component| {
                path.push(component.to_owned());
            });
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
            clean::ModuleItem(..) => write!(fmt.buf, "Module "),
            clean::FunctionItem(..) => write!(fmt.buf, "Function "),
            clean::TraitItem(..) => write!(fmt.buf, "Trait "),
            clean::StructItem(..) => write!(fmt.buf, "Struct "),
            clean::EnumItem(..) => write!(fmt.buf, "Enum "),
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
        clean::ModuleItem(..) => *item.name.get_ref() + "/index.html",
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

fn document(w: &mut Writer, item: &clean::Item) {
    match item.doc_value() {
        Some(s) => {
            write!(w, "<div class='docblock'>{}</div>", Markdown(s));
        }
        None => {}
    }
}

fn item_module(w: &mut Writer, cx: &Context,
               item: &clean::Item, items: &[clean::Item]) {
    document(w, item);
    debug!("{:?}", items);
    let mut indices = vec::from_fn(items.len(), |i| i);

    fn cmp(i1: &clean::Item, i2: &clean::Item, idx1: uint, idx2: uint) -> Ordering {
        if shortty(i1) == shortty(i2) {
            return i1.name.cmp(&i2.name);
        }
        match (&i1.inner, &i2.inner) {
            (&clean::ViewItemItem(ref a), &clean::ViewItemItem(ref b)) => {
                match (&a.inner, &b.inner) {
                    (&clean::ExternMod(..), _) => Less,
                    (_, &clean::ExternMod(..)) => Greater,
                    _ => idx1.cmp(&idx2),
                }
            }
            (&clean::ViewItemItem(..), _) => Less,
            (_, &clean::ViewItemItem(..)) => Greater,
            (&clean::ModuleItem(..), _) => Less,
            (_, &clean::ModuleItem(..)) => Greater,
            (&clean::StructItem(..), _) => Less,
            (_, &clean::StructItem(..)) => Greater,
            (&clean::EnumItem(..), _) => Less,
            (_, &clean::EnumItem(..)) => Greater,
            (&clean::StaticItem(..), _) => Less,
            (_, &clean::StaticItem(..)) => Greater,
            (&clean::ForeignFunctionItem(..), _) => Less,
            (_, &clean::ForeignFunctionItem(..)) => Greater,
            (&clean::ForeignStaticItem(..), _) => Less,
            (_, &clean::ForeignStaticItem(..)) => Greater,
            (&clean::TraitItem(..), _) => Less,
            (_, &clean::TraitItem(..)) => Greater,
            (&clean::FunctionItem(..), _) => Less,
            (_, &clean::FunctionItem(..)) => Greater,
            (&clean::TypedefItem(..), _) => Less,
            (_, &clean::TypedefItem(..)) => Greater,
            _ => idx1.cmp(&idx2),
        }
    }

    debug!("{:?}", indices);
    indices.sort_by(|&i1, &i2| cmp(&items[i1], &items[i2], i1, i2));

    debug!("{:?}", indices);
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
                clean::ModuleItem(..)          => "Modules",
                clean::StructItem(..)          => "Structs",
                clean::EnumItem(..)            => "Enums",
                clean::FunctionItem(..)        => "Functions",
                clean::TypedefItem(..)         => "Type Definitions",
                clean::StaticItem(..)          => "Statics",
                clean::TraitItem(..)           => "Traits",
                clean::ImplItem(..)            => "Implementations",
                clean::ViewItemItem(..)        => "Reexports",
                clean::TyMethodItem(..)        => "Type Methods",
                clean::MethodItem(..)          => "Methods",
                clean::StructFieldItem(..)     => "Struct Fields",
                clean::VariantItem(..)         => "Variants",
                clean::ForeignFunctionItem(..) => "Foreign Functions",
                clean::ForeignStaticItem(..)   => "Foreign Statics",
            });
        }

        match myitem.inner {
            clean::StaticItem(ref s) | clean::ForeignStaticItem(ref s) => {
                struct Initializer<'a>(&'a str);
                impl<'a> fmt::Show for Initializer<'a> {
                    fn fmt(s: &Initializer<'a>, f: &mut fmt::Formatter) {
                        let Initializer(s) = *s;
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
                    clean::ExternMod(ref name, ref src, _) => {
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

fn item_function(w: &mut Writer, it: &clean::Item, f: &clean::Function) {
    write!(w, "<pre class='fn'>{vis}{purity}fn {name}{generics}{decl}</pre>",
           vis = VisSpace(it.visibility),
           purity = PuritySpace(f.purity),
           name = it.name.get_ref().as_slice(),
           generics = f.generics,
           decl = f.decl);
    document(w, it);
}

fn item_trait(w: &mut Writer, it: &clean::Item, t: &clean::Trait) {
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

    fn meth(w: &mut Writer, m: &clean::TraitMethod) {
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

    local_data::get(cache_key, |cache| {
        let cache = cache.unwrap().get();
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
    })
}

fn render_method(w: &mut Writer, meth: &clean::Item, withlink: bool) {
    fn fun(w: &mut Writer, it: &clean::Item, purity: ast::Purity,
           g: &clean::Generics, selfty: &clean::SelfTy, d: &clean::FnDecl,
           withlink: bool) {
        write!(w, "{}fn {withlink, select,
                            true{<a href='\\#{ty}.{name}'
                                    class='fnname'>{name}</a>}
                            other{<span class='fnname'>{name}</span>}
                        }{generics}{decl}",
               match purity {
                   ast::UnsafeFn => "unsafe ",
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

fn item_struct(w: &mut Writer, it: &clean::Item, s: &clean::Struct) {
    write!(w, "<pre class='struct'>");
    render_struct(w, it, Some(&s.generics), s.struct_type, s.fields,
                  s.fields_stripped, "", true);
    write!(w, "</pre>");

    document(w, it);
    match s.struct_type {
        doctree::Plain if s.fields.len() > 0 => {
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

fn item_enum(w: &mut Writer, it: &clean::Item, e: &clean::Enum) {
    write!(w, "<pre class='enum'>{}enum {}{}",
           VisSpace(it.visibility),
           it.name.get_ref().as_slice(),
           e.generics);
    if e.variants.len() == 0 && !e.variants_stripped {
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
                                          s.fields_stripped, "    ", false);
                        }
                    }
                }
                _ => unreachable!()
            }
            write!(w, ",\n");
        }

        if e.variants_stripped {
            write!(w, "    // some variants omitted\n");
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
            match variant.inner {
                clean::VariantItem(ref var) => {
                    match var.kind {
                        clean::StructVariant(ref s) => {
                            write!(w, "<h3 class='fields'>Fields</h3>\n<table>");
                            for field in s.fields.iter() {
                                write!(w, "<tr><td id='variant.{v}.field.{f}'>\
                                           <code>{f}</code></td><td>",
                                       v = variant.name.get_ref().as_slice(),
                                       f = field.name.get_ref().as_slice());
                                document(w, field);
                                write!(w, "</td></tr>");
                            }
                            write!(w, "</table>");
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            write!(w, "</td></tr>");
        }
        write!(w, "</table>");

    }
    render_methods(w, it);
}

fn render_struct(w: &mut Writer, it: &clean::Item,
                 g: Option<&clean::Generics>,
                 ty: doctree::StructType,
                 fields: &[clean::Item],
                 fields_stripped: bool,
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

            if fields_stripped {
                write!(w, "    // some fields omitted\n{}", tab);
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

fn render_methods(w: &mut Writer, it: &clean::Item) {
    local_data::get(cache_key, |cache| {
        let c = cache.unwrap().get();
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
    })
}

fn render_impl(w: &mut Writer, i: &clean::Impl, dox: &Option<~str>) {
    write!(w, "<h3 class='impl'><code>impl{} ", i.generics);
    let trait_id = match i.trait_ {
        Some(ref ty) => {
            write!(w, "{} for ", *ty);
            match *ty {
                clean::ResolvedPath { id, .. } => Some(id),
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

    fn docmeth(w: &mut Writer, item: &clean::Item) -> bool {
        write!(w, "<h4 id='method.{}' class='method'><code>",
               *item.name.get_ref());
        render_method(w, item, false);
        write!(w, "</code></h4>\n");
        match item.doc_value() {
            Some(s) => {
                write!(w, "<div class='docblock'>{}</div>", Markdown(s));
                true
            }
            None => false
        }
    }

    write!(w, "<div class='methods'>");
    for meth in i.methods.iter() {
        if docmeth(w, meth) {
            continue
        }

        // No documentation? Attempt to slurp in the trait's documentation
        let trait_id = match trait_id {
            None => continue,
            Some(id) => id,
        };
        local_data::get(cache_key, |cache| {
            let cache = cache.unwrap().get();
            match cache.traits.find(&trait_id) {
                Some(t) => {
                    let name = meth.name.clone();
                    match t.methods.iter().find(|t| t.item().name == name) {
                        Some(method) => {
                            match method.item().doc_value() {
                                Some(s) => {
                                    write!(w,
                                           "<div class='docblock'>{}</div>",
                                           Markdown(s));
                                }
                                None => {}
                            }
                        }
                        None => {}
                    }
                }
                None => {}
            }
        })
    }

    // If we've implemented a trait, then also emit documentation for all
    // default methods which weren't overridden in the implementation block.
    match trait_id {
        None => {}
        Some(id) => {
            local_data::get(cache_key, |cache| {
                let cache = cache.unwrap().get();
                match cache.traits.find(&id) {
                    Some(t) => {
                        for method in t.methods.iter() {
                            let n = method.item().name.clone();
                            match i.methods.iter().find(|m| m.name == n) {
                                Some(..) => continue,
                                None => {}
                            }

                            docmeth(w, method.item());
                        }
                    }
                    None => {}
                }
            })
        }
    }
    write!(w, "</div>");
}

fn item_typedef(w: &mut Writer, it: &clean::Item, t: &clean::Typedef) {
    write!(w, "<pre class='typedef'>type {}{} = {};</pre>",
           it.name.get_ref().as_slice(),
           t.generics,
           t.type_);

    document(w, it);
}

impl<'a> fmt::Show for Sidebar<'a> {
    fn fmt(s: &Sidebar<'a>, fmt: &mut fmt::Formatter) {
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

        fn block(w: &mut Writer, short: &str, longty: &str,
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
        items.sort();
    }
    return map;
}

impl<'a> fmt::Show for Source<'a> {
    fn fmt(s: &Source<'a>, fmt: &mut fmt::Formatter) {
        let Source(s) = *s;
        let lines = s.lines().len();
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
