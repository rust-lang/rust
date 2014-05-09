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

use collections::{HashMap, HashSet};
use std::fmt;
use std::io::{fs, File, BufferedWriter, MemWriter, BufferedReader};
use std::io;
use std::str;
use std::strbuf::StrBuf;

use sync::Arc;
use serialize::json::ToJson;
use syntax::ast;
use syntax::attr;
use syntax::parse::token::InternedString;
use rustc::util::nodemap::NodeSet;

use clean;
use doctree;
use fold::DocFolder;
use html::item_type;
use html::item_type::{ItemType, shortty};
use html::format::{VisSpace, Method, FnStyleSpace};
use html::layout;
use html::markdown;
use html::markdown::Markdown;
use html::highlight;

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
    pub current: Vec<~str> ,
    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    pub root_path: StrBuf,
    /// The current destination folder of where HTML artifacts should be placed.
    /// This changes as the context descends into the module hierarchy.
    pub dst: Path,
    /// This describes the layout of each page, and is not modified after
    /// creation of the context (contains info like the favicon)
    pub layout: layout::Layout,
    /// This map is a list of what should be displayed on the sidebar of the
    /// current page. The key is the section header (traits, modules,
    /// functions), and the value is the list of containers belonging to this
    /// header. This map will change depending on the surrounding context of the
    /// page.
    pub sidebar: HashMap<~str, Vec<~str> >,
    /// This flag indicates whether [src] links should be generated or not. If
    /// the source files are present in the html rendering, then this will be
    /// `true`.
    pub include_sources: bool,
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
pub enum Implementor {
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
/// to `Send` so it may be stored in a `Arc` instance and shared among the various
/// rendering tasks.
pub struct Cache {
    /// Mapping of typaram ids to the name of the type parameter. This is used
    /// when pretty-printing a type (so pretty printing doesn't have to
    /// painfully maintain a context like this)
    pub typarams: HashMap<ast::NodeId, ~str>,

    /// Maps a type id to all known implementations for that type. This is only
    /// recognized for intra-crate `ResolvedPath` types, and is used to print
    /// out extra documentation on the page of an enum/struct.
    ///
    /// The values of the map are a list of implementations and documentation
    /// found on that implementation.
    pub impls: HashMap<ast::NodeId, Vec<(clean::Impl, Option<~str>)> >,

    /// Maintains a mapping of local crate node ids to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    pub paths: HashMap<ast::NodeId, (Vec<~str> , ItemType)>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    pub traits: HashMap<ast::NodeId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    pub implementors: HashMap<ast::NodeId, Vec<Implementor> >,

    /// Cache of where external crate documentation can be found.
    pub extern_locations: HashMap<ast::CrateNum, ExternalLocation>,

    // Private fields only used when initially crawling a crate to build a cache

    stack: Vec<~str> ,
    parent_stack: Vec<ast::NodeId> ,
    search_index: Vec<IndexItem> ,
    privmod: bool,
    public_items: NodeSet,

    // In rare case where a structure is defined in one module but implemented
    // in another, if the implementing module is parsed before defining module,
    // then the fully qualified name of the structure isn't presented in `paths`
    // yet when its implementation methods are being indexed. Caches such methods
    // and their parent id here and indexes them at the end of crate parsing.
    orphan_methods: Vec<(ast::NodeId, clean::Item)>,
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
    ty: ItemType,
    name: ~str,
    path: ~str,
    desc: ~str,
    parent: Option<ast::NodeId>,
}

// TLS keys used to carry information around during rendering.

local_data_key!(pub cache_key: Arc<Cache>)
local_data_key!(pub current_location_key: Vec<~str> )

/// Generates the documentation for `crate` into the directory `dst`
pub fn run(mut krate: clean::Crate, dst: Path) -> io::IoResult<()> {
    let mut cx = Context {
        dst: dst,
        current: Vec::new(),
        root_path: StrBuf::new(),
        sidebar: HashMap::new(),
        layout: layout::Layout {
            logo: "".to_owned(),
            favicon: "".to_owned(),
            krate: krate.name.clone(),
        },
        include_sources: true,
    };
    try!(mkdir(&cx.dst));

    match krate.module.as_ref().map(|m| m.doc_list().unwrap_or(&[])) {
        Some(attrs) => {
            for attr in attrs.iter() {
                match *attr {
                    clean::NameValue(ref x, ref s) if "html_favicon_url" == *x => {
                        cx.layout.favicon = s.to_owned();
                    }
                    clean::NameValue(ref x, ref s) if "html_logo_url" == *x => {
                        cx.layout.logo = s.to_owned();
                    }
                    clean::Word(ref x) if "html_no_source" == *x => {
                        cx.include_sources = false;
                    }
                    _ => {}
                }
            }
        }
        None => {}
    }

    // Crawl the crate to build various caches used for the output
    let public_items = ::analysiskey.get().map(|a| a.public_items.clone());
    let public_items = public_items.unwrap_or(NodeSet::new());
    let mut cache = Cache {
        impls: HashMap::new(),
        typarams: HashMap::new(),
        paths: HashMap::new(),
        traits: HashMap::new(),
        implementors: HashMap::new(),
        stack: Vec::new(),
        parent_stack: Vec::new(),
        search_index: Vec::new(),
        extern_locations: HashMap::new(),
        privmod: false,
        public_items: public_items,
        orphan_methods: Vec::new(),
    };
    cache.stack.push(krate.name.clone());
    krate = cache.fold_crate(krate);

    let mut nodeid_to_pathid = HashMap::new();
    let mut pathid_to_nodeid = Vec::new();
    {
        let Cache { search_index: ref mut index,
                    orphan_methods: ref meths, paths: ref mut paths, ..} = cache;

        // Attach all orphan methods to the type's definition if the type
        // has since been learned.
        for &(ref pid, ref item) in meths.iter() {
            match paths.find(pid) {
                Some(&(ref fqp, _)) => {
                    index.push(IndexItem {
                        ty: shortty(item),
                        name: item.name.clone().unwrap(),
                        path: fqp.slice_to(fqp.len() - 1).connect("::"),
                        desc: shorter(item.doc_value()).to_owned(),
                        parent: Some(*pid),
                    });
                },
                None => {}
            }
        };

        // Reduce `NodeId` in paths into smaller sequential numbers,
        // and prune the paths that do not appear in the index.
        for item in index.iter() {
            match item.parent {
                Some(nodeid) => {
                    if !nodeid_to_pathid.contains_key(&nodeid) {
                        let pathid = pathid_to_nodeid.len();
                        nodeid_to_pathid.insert(nodeid, pathid);
                        pathid_to_nodeid.push(nodeid);
                    }
                }
                None => {}
            }
        }
        assert_eq!(nodeid_to_pathid.len(), pathid_to_nodeid.len());
    }

    // Publish the search index
    let index = {
        let mut w = MemWriter::new();
        try!(write!(&mut w, r#"searchIndex['{}'] = \{"items":["#, krate.name));

        let mut lastpath = "".to_owned();
        for (i, item) in cache.search_index.iter().enumerate() {
            // Omit the path if it is same to that of the prior item.
            let path;
            if lastpath == item.path {
                path = "";
            } else {
                lastpath = item.path.clone();
                path = item.path.as_slice();
            };

            if i > 0 {
                try!(write!(&mut w, ","));
            }
            try!(write!(&mut w, r#"[{:u},"{}","{}",{}"#,
                        item.ty, item.name, path,
                        item.desc.to_json().to_str()));
            match item.parent {
                Some(nodeid) => {
                    let pathid = *nodeid_to_pathid.find(&nodeid).unwrap();
                    try!(write!(&mut w, ",{}", pathid));
                }
                None => {}
            }
            try!(write!(&mut w, "]"));
        }

        try!(write!(&mut w, r#"],"paths":["#));

        for (i, &nodeid) in pathid_to_nodeid.iter().enumerate() {
            let &(ref fqp, short) = cache.paths.find(&nodeid).unwrap();
            if i > 0 {
                try!(write!(&mut w, ","));
            }
            try!(write!(&mut w, r#"[{:u},"{}"]"#,
                        short, *fqp.last().unwrap()));
        }

        try!(write!(&mut w, r"]\};"));

        str::from_utf8(w.unwrap().as_slice()).unwrap().to_owned()
    };

    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    {
        try!(mkdir(&cx.dst));
        let _lock = ::flock::Lock::new(&cx.dst.join(".lock"));

        // Add all the static files. These may already exist, but we just
        // overwrite them anyway to make sure that they're fresh and up-to-date.
        try!(write(cx.dst.join("jquery.js"),
                   include_bin!("static/jquery-2.1.0.min.js")));
        try!(write(cx.dst.join("main.js"), include_bin!("static/main.js")));
        try!(write(cx.dst.join("main.css"), include_bin!("static/main.css")));
        try!(write(cx.dst.join("normalize.css"),
                   include_bin!("static/normalize.css")));
        try!(write(cx.dst.join("FiraSans-Regular.woff"),
                   include_bin!("static/FiraSans-Regular.woff")));
        try!(write(cx.dst.join("FiraSans-Medium.woff"),
                   include_bin!("static/FiraSans-Medium.woff")));
        try!(write(cx.dst.join("Heuristica-Regular.woff"),
                   include_bin!("static/Heuristica-Regular.woff")));
        try!(write(cx.dst.join("Heuristica-Italic.woff"),
                   include_bin!("static/Heuristica-Italic.woff")));
        try!(write(cx.dst.join("Heuristica-Bold.woff"),
                   include_bin!("static/Heuristica-Bold.woff")));

        // Update the search index
        let dst = cx.dst.join("search-index.js");
        let mut all_indexes = Vec::new();
        all_indexes.push(index);
        if dst.exists() {
            for line in BufferedReader::new(File::open(&dst)).lines() {
                let line = try!(line);
                if !line.starts_with("searchIndex") { continue }
                if line.starts_with(format!("searchIndex['{}']", krate.name)) {
                    continue
                }
                all_indexes.push(line);
            }
        }
        let mut w = try!(File::create(&dst));
        try!(writeln!(&mut w, r"var searchIndex = \{\};"));
        for index in all_indexes.iter() {
            try!(writeln!(&mut w, "{}", *index));
        }
        try!(writeln!(&mut w, "initSearch(searchIndex);"));
    }

    // Render all source files (this may turn into a giant no-op)
    {
        info!("emitting source files");
        let dst = cx.dst.join("src");
        try!(mkdir(&dst));
        let dst = dst.join(krate.name.as_slice());
        try!(mkdir(&dst));
        let mut folder = SourceCollector {
            dst: dst,
            seen: HashSet::new(),
            cx: &mut cx,
        };
        krate = folder.fold_crate(krate);
    }

    for &(n, ref e) in krate.externs.iter() {
        cache.extern_locations.insert(n, extern_location(e, &cx.dst));
    }

    // And finally render the whole crate's documentation
    cx.krate(krate, cache)
}

/// Writes the entire contents of a string to a destination, not attempting to
/// catch any errors.
fn write(dst: Path, contents: &[u8]) -> io::IoResult<()> {
    File::create(&dst).write(contents)
}

/// Makes a directory on the filesystem, failing the task if an error occurs and
/// skipping if the directory already exists.
fn mkdir(path: &Path) -> io::IoResult<()> {
    if !path.exists() {
        fs::mkdir(path, io::UserRWX)
    } else {
        Ok(())
    }
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
            clean::List(ref x, ref list) if "doc" == *x => {
                for attr in list.iter() {
                    match *attr {
                        clean::NameValue(ref x, ref s) if "html_root_url" == *x => {
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
            self.cx.include_sources = match self.emit_source(item.source.filename) {
                Ok(()) => true,
                Err(e) => {
                    println!("warning: source code was requested to be rendered, \
                              but processing `{}` had an error: {}",
                             item.source.filename, e);
                    println!("         skipping rendering of source code");
                    false
                }
            };
            self.seen.insert(item.source.filename.clone());
        }

        self.fold_item_recur(item)
    }
}

impl<'a> SourceCollector<'a> {
    /// Renders the given filename into its corresponding HTML source file.
    fn emit_source(&mut self, filename: &str) -> io::IoResult<()> {
        let p = Path::new(filename);

        // If we couldn't open this file, then just returns because it
        // probably means that it's some standard library macro thing and we
        // can't have the source to it anyway.
        let contents = match File::open(&p).read_to_end() {
            Ok(r) => r,
            // macros from other libraries get special filenames which we can
            // safely ignore
            Err(..) if filename.starts_with("<") &&
                       filename.ends_with("macros>") => return Ok(()),
            Err(e) => return Err(e)
        };
        let contents = str::from_utf8(contents.as_slice()).unwrap();

        // Remove the utf-8 BOM if any
        let contents = if contents.starts_with("\ufeff") {
            contents.as_slice().slice_from(3)
        } else {
            contents.as_slice()
        };

        // Create the intermediate directories
        let mut cur = self.dst.clone();
        let mut root_path = StrBuf::from_str("../../");
        clean_srcpath(p.dirname(), |component| {
            cur.push(component);
            mkdir(&cur).unwrap();
            root_path.push_str("../");
        });

        cur.push(Vec::from_slice(p.filename().expect("source has no filename"))
                 .append(bytes!(".html")));
        let mut w = BufferedWriter::new(try!(File::create(&cur)));

        let title = format!("{} -- source", cur.filename_display());
        let page = layout::Page {
            title: title,
            ty: "source",
            root_path: root_path.as_slice(),
        };
        try!(layout::render(&mut w as &mut Writer, &self.cx.layout,
                              &page, &(""), &Source(contents)));
        try!(w.flush());
        return Ok(());
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
                            Vec::new()
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
                        (Some(*self.parent_stack.last().unwrap()),
                         Some(self.stack.slice_to(self.stack.len() - 1)))

                    }
                    clean::MethodItem(..) => {
                        if self.parent_stack.len() == 0 {
                            (None, None)
                        } else {
                            let last = self.parent_stack.last().unwrap();
                            let path = match self.paths.find(last) {
                                Some(&(_, item_type::Trait)) =>
                                    Some(self.stack.slice_to(self.stack.len() - 1)),
                                // The current stack not necessarily has correlation for
                                // where the type was defined. On the other hand,
                                // `paths` always has the right information if present.
                                Some(&(ref fqp, item_type::Struct)) |
                                Some(&(ref fqp, item_type::Enum)) =>
                                    Some(fqp.slice_to(fqp.len() - 1)),
                                Some(..) => Some(self.stack.as_slice()),
                                None => None
                            };
                            (Some(*last), path)
                        }
                    }
                    _ => (None, Some(self.stack.as_slice()))
                };
                match parent {
                    (parent, Some(path)) if !self.privmod => {
                        self.search_index.push(IndexItem {
                            ty: shortty(&item),
                            name: s.to_owned(),
                            path: path.connect("::"),
                            desc: shorter(item.doc_value()).to_owned(),
                            parent: parent,
                        });
                    }
                    (Some(parent), None) if !self.privmod => {
                        // We have a parent, but we don't know where they're
                        // defined yet. Wait for later to index this item.
                        self.orphan_methods.push((parent, item.clone()))
                    }
                    _ => {}
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
            clean::ForeignFunctionItem(..) => {
                // Reexported items mean that the same id can show up twice in
                // the rustdoc ast that we're looking at. We know, however, that
                // a reexported item doesn't show up in the `public_items` map,
                // so we can skip inserting into the paths map if there was
                // already an entry present and we're not a public item.
                if !self.paths.contains_key(&item.id) ||
                   self.public_items.contains(&item.id) {
                    self.paths.insert(item.id,
                                      (self.stack.clone(), shortty(&item)));
                }
            }
            // link variants to their parent enum because pages aren't emitted
            // for each variant
            clean::VariantItem(..) => {
                let mut stack = self.stack.clone();
                stack.pop();
                self.paths.insert(item.id, (stack, item_type::Enum));
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
                                    Vec::new()
                                });
                                // extract relevant documentation for this impl
                                match attrs.move_iter().find(|a| {
                                    match *a {
                                        clean::NameValue(ref x, _) if "doc" == *x => true,
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

        mkdir(&self.dst).unwrap();
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
    fn krate(self, mut krate: clean::Crate, cache: Cache) -> io::IoResult<()> {
        let mut item = match krate.module.take() {
            Some(i) => i,
            None => return Ok(())
        };
        item.name = Some(krate.name);

        // using a rwarc makes this parallelizable in the future
        cache_key.replace(Some(Arc::new(cache)));

        let mut work = vec!((self, item));
        loop {
            match work.pop() {
                Some((mut cx, item)) => try!(cx.item(item, |cx, item| {
                    work.push((cx.clone(), item));
                })),
                None => break,
            }
        }
        Ok(())
    }

    /// Non-parellelized version of rendering an item. This will take the input
    /// item, render its contents, and then invoke the specified closure with
    /// all sub-items which need to be rendered.
    ///
    /// The rendering driver uses this closure to queue up more work.
    fn item(&mut self, item: clean::Item,
            f: |&mut Context, clean::Item|) -> io::IoResult<()> {
        fn render(w: io::File, cx: &mut Context, it: &clean::Item,
                  pushname: bool) -> io::IoResult<()> {
            info!("Rendering an item to {}", w.path().display());
            // A little unfortunate that this is done like this, but it sure
            // does make formatting *a lot* nicer.
            current_location_key.replace(Some(cx.current.clone()));

            let mut title = StrBuf::from_str(cx.current.connect("::"));
            if pushname {
                if title.len() > 0 {
                    title.push_str("::");
                }
                title.push_str(*it.name.get_ref());
            }
            title.push_str(" - Rust");
            let page = layout::Page {
                ty: shortty(it).to_static_str(),
                root_path: cx.root_path.as_slice(),
                title: title.as_slice(),
            };

            markdown::reset_headers();

            // We have a huge number of calls to write, so try to alleviate some
            // of the pain by using a buffered writer instead of invoking the
            // write sycall all the time.
            let mut writer = BufferedWriter::new(w);
            try!(layout::render(&mut writer as &mut Writer, &cx.layout, &page,
                                  &Sidebar{ cx: cx, item: it },
                                  &Item{ cx: cx, item: it }));
            writer.flush()
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
                    let dst = try!(File::create(&dst));
                    try!(render(dst, this, &item, false));

                    let m = match item.inner {
                        clean::ModuleItem(m) => m,
                        _ => unreachable!()
                    };
                    this.sidebar = build_sidebar(&m);
                    for item in m.items.move_iter() {
                        f(this,item);
                    }
                    Ok(())
                })
            }

            // Things which don't have names (like impls) don't get special
            // pages dedicated to them.
            _ if item.name.is_some() => {
                let dst = self.dst.join(item_path(&item));
                let dst = try!(File::create(&dst));
                render(dst, self, &item, true)
            }

            _ => Ok(())
        }
    }
}

impl<'a> Item<'a> {
    fn ismodule(&self) -> bool {
        match self.item.inner {
            clean::ModuleItem(..) => true, _ => false
        }
    }

    fn link(&self) -> ~str {
        let mut path = Vec::new();
        clean_srcpath(self.item.source.filename.as_bytes(), |component| {
            path.push(component.to_owned());
        });
        let href = if self.item.source.loline == self.item.source.hiline {
            format!("{}", self.item.source.loline)
        } else {
            format!("{}-{}", self.item.source.loline, self.item.source.hiline)
        };
        format!("{root}src/{krate}/{path}.html\\#{href}",
                root = self.cx.root_path,
                krate = self.cx.layout.krate,
                path = path.connect("/"),
                href = href)
    }
}

impl<'a> fmt::Show for Item<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        // Write the breadcrumb trail header for the top
        try!(write!(fmt.buf, "\n<h1 class='fqn'>"));
        match self.item.inner {
            clean::ModuleItem(ref m) => if m.is_crate {
                    try!(write!(fmt.buf, "Crate "));
                } else {
                    try!(write!(fmt.buf, "Module "));
                },
            clean::FunctionItem(..) => try!(write!(fmt.buf, "Function ")),
            clean::TraitItem(..) => try!(write!(fmt.buf, "Trait ")),
            clean::StructItem(..) => try!(write!(fmt.buf, "Struct ")),
            clean::EnumItem(..) => try!(write!(fmt.buf, "Enum ")),
            _ => {}
        }
        let cur = self.cx.current.as_slice();
        let amt = if self.ismodule() { cur.len() - 1 } else { cur.len() };
        for (i, component) in cur.iter().enumerate().take(amt) {
            let mut trail = StrBuf::new();
            for _ in range(0, cur.len() - i - 1) {
                trail.push_str("../");
            }
            try!(write!(fmt.buf, "<a href='{}index.html'>{}</a>::",
                          trail, component.as_slice()));
        }
        try!(write!(fmt.buf, "<a class='{}' href=''>{}</a>",
                      shortty(self.item), self.item.name.get_ref().as_slice()));

        // Write stability attributes
        match attr::find_stability(self.item.attrs.iter()) {
            Some(ref stability) => {
                try!(write!(fmt.buf,
                       "<a class='stability {lvl}' title='{reason}'>{lvl}</a>",
                       lvl = stability.level.to_str(),
                       reason = match stability.text {
                           Some(ref s) => (*s).clone(),
                           None => InternedString::new(""),
                       }));
            }
            None => {}
        }

        // Write `src` tag
        if self.cx.include_sources {
            try!(write!(fmt.buf, "<a class='source' href='{}'>[src]</a>",
                        self.link()));
        }
        try!(write!(fmt.buf, "</h1>\n"));

        match self.item.inner {
            clean::ModuleItem(ref m) => {
                item_module(fmt.buf, self.cx, self.item, m.items.as_slice())
            }
            clean::FunctionItem(ref f) | clean::ForeignFunctionItem(ref f) =>
                item_function(fmt.buf, self.item, f),
            clean::TraitItem(ref t) => item_trait(fmt.buf, self.item, t),
            clean::StructItem(ref s) => item_struct(fmt.buf, self.item, s),
            clean::EnumItem(ref e) => item_enum(fmt.buf, self.item, e),
            clean::TypedefItem(ref t) => item_typedef(fmt.buf, self.item, t),
            clean::MacroItem(ref m) => item_macro(fmt.buf, self.item, m),
            _ => Ok(())
        }
    }
}

fn item_path(item: &clean::Item) -> ~str {
    match item.inner {
        clean::ModuleItem(..) => *item.name.get_ref() + "/index.html",
        _ => shortty(item).to_static_str() + "." + *item.name.get_ref() + ".html"
    }
}

fn full_path(cx: &Context, item: &clean::Item) -> ~str {
    let mut s = StrBuf::from_str(cx.current.connect("::"));
    s.push_str("::");
    s.push_str(item.name.get_ref().as_slice());
    return s.into_owned();
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

fn document(w: &mut Writer, item: &clean::Item) -> fmt::Result {
    match item.doc_value() {
        Some(s) => {
            try!(write!(w, "<div class='docblock'>{}</div>", Markdown(s)));
        }
        None => {}
    }
    Ok(())
}

fn item_module(w: &mut Writer, cx: &Context,
               item: &clean::Item, items: &[clean::Item]) -> fmt::Result {
    try!(document(w, item));
    debug!("{:?}", items);
    let mut indices = Vec::from_fn(items.len(), |i| i);

    fn cmp(i1: &clean::Item, i2: &clean::Item, idx1: uint, idx2: uint) -> Ordering {
        if shortty(i1) == shortty(i2) {
            return i1.name.cmp(&i2.name);
        }
        match (&i1.inner, &i2.inner) {
            (&clean::ViewItemItem(ref a), &clean::ViewItemItem(ref b)) => {
                match (&a.inner, &b.inner) {
                    (&clean::ExternCrate(..), _) => Less,
                    (_, &clean::ExternCrate(..)) => Greater,
                    _ => idx1.cmp(&idx2),
                }
            }
            (&clean::ViewItemItem(..), _) => Less,
            (_, &clean::ViewItemItem(..)) => Greater,
            (&clean::ModuleItem(..), _) => Less,
            (_, &clean::ModuleItem(..)) => Greater,
            (&clean::MacroItem(..), _) => Less,
            (_, &clean::MacroItem(..)) => Greater,
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
    let mut curty = None;
    for &idx in indices.iter() {
        let myitem = &items[idx];

        let myty = Some(shortty(myitem));
        if myty != curty {
            if curty.is_some() {
                try!(write!(w, "</table>"));
            }
            curty = myty;
            let (short, name) = match myitem.inner {
                clean::ModuleItem(..)          => ("modules", "Modules"),
                clean::StructItem(..)          => ("structs", "Structs"),
                clean::EnumItem(..)            => ("enums", "Enums"),
                clean::FunctionItem(..)        => ("functions", "Functions"),
                clean::TypedefItem(..)         => ("types", "Type Definitions"),
                clean::StaticItem(..)          => ("statics", "Statics"),
                clean::TraitItem(..)           => ("traits", "Traits"),
                clean::ImplItem(..)            => ("impls", "Implementations"),
                clean::ViewItemItem(..)        => ("reexports", "Reexports"),
                clean::TyMethodItem(..)        => ("tymethods", "Type Methods"),
                clean::MethodItem(..)          => ("methods", "Methods"),
                clean::StructFieldItem(..)     => ("fields", "Struct Fields"),
                clean::VariantItem(..)         => ("variants", "Variants"),
                clean::ForeignFunctionItem(..) => ("ffi-fns", "Foreign Functions"),
                clean::ForeignStaticItem(..)   => ("ffi-statics", "Foreign Statics"),
                clean::MacroItem(..)           => ("macros", "Macros"),
            };
            try!(write!(w,
                        "<h2 id='{id}' class='section-header'>\
                        <a href=\"\\#{id}\">{name}</a></h2>\n<table>",
                        id = short, name = name));
        }

        match myitem.inner {
            clean::StaticItem(ref s) | clean::ForeignStaticItem(ref s) => {
                struct Initializer<'a>(&'a str, Item<'a>);
                impl<'a> fmt::Show for Initializer<'a> {
                    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        let Initializer(s, item) = *self;
                        if s.len() == 0 { return Ok(()); }
                        try!(write!(f.buf, "<code> = </code>"));
                        if s.contains("\n") {
                            write!(f.buf,
                                   "<a href='{}'>[definition]</a>",
                                   item.link())
                        } else {
                            write!(f.buf, "<code>{}</code>", s.as_slice())
                        }
                    }
                }

                try!(write!(w, "
                    <tr>
                        <td><code>{}static {}: {}</code>{}</td>
                        <td class='docblock'>{}&nbsp;</td>
                    </tr>
                ",
                VisSpace(myitem.visibility),
                *myitem.name.get_ref(),
                s.type_,
                Initializer(s.expr, Item { cx: cx, item: myitem }),
                Markdown(blank(myitem.doc_value()))));
            }

            clean::ViewItemItem(ref item) => {
                match item.inner {
                    clean::ExternCrate(ref name, ref src, _) => {
                        try!(write!(w, "<tr><td><code>extern crate {}",
                                      name.as_slice()));
                        match *src {
                            Some(ref src) => try!(write!(w, " = \"{}\"",
                                                           src.as_slice())),
                            None => {}
                        }
                        try!(write!(w, ";</code></td></tr>"));
                    }

                    clean::Import(ref import) => {
                        try!(write!(w, "<tr><td><code>{}{}</code></td></tr>",
                                      VisSpace(myitem.visibility),
                                      *import));
                    }
                }

            }

            _ => {
                if myitem.name.is_none() { continue }
                try!(write!(w, "
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
                title = full_path(cx, myitem)));
            }
        }
    }
    write!(w, "</table>")
}

fn item_function(w: &mut Writer, it: &clean::Item,
                 f: &clean::Function) -> fmt::Result {
    try!(write!(w, "<pre class='rust fn'>{vis}{fn_style}fn \
                    {name}{generics}{decl}</pre>",
           vis = VisSpace(it.visibility),
           fn_style = FnStyleSpace(f.fn_style),
           name = it.name.get_ref().as_slice(),
           generics = f.generics,
           decl = f.decl));
    document(w, it)
}

fn item_trait(w: &mut Writer, it: &clean::Item,
              t: &clean::Trait) -> fmt::Result {
    let mut parents = StrBuf::new();
    if t.parents.len() > 0 {
        parents.push_str(": ");
        for (i, p) in t.parents.iter().enumerate() {
            if i > 0 { parents.push_str(" + "); }
            parents.push_str(format!("{}", *p));
        }
    }

    // Output the trait definition
    try!(write!(w, "<pre class='rust trait'>{}trait {}{}{} ",
                  VisSpace(it.visibility),
                  it.name.get_ref().as_slice(),
                  t.generics,
                  parents));
    let required = t.methods.iter().filter(|m| m.is_req()).collect::<Vec<&clean::TraitMethod>>();
    let provided = t.methods.iter().filter(|m| !m.is_req()).collect::<Vec<&clean::TraitMethod>>();

    if t.methods.len() == 0 {
        try!(write!(w, "\\{ \\}"));
    } else {
        try!(write!(w, "\\{\n"));
        for m in required.iter() {
            try!(write!(w, "    "));
            try!(render_method(w, m.item()));
            try!(write!(w, ";\n"));
        }
        if required.len() > 0 && provided.len() > 0 {
            try!(w.write("\n".as_bytes()));
        }
        for m in provided.iter() {
            try!(write!(w, "    "));
            try!(render_method(w, m.item()));
            try!(write!(w, " \\{ ... \\}\n"));
        }
        try!(write!(w, "\\}"));
    }
    try!(write!(w, "</pre>"));

    // Trait documentation
    try!(document(w, it));

    fn meth(w: &mut Writer, m: &clean::TraitMethod) -> fmt::Result {
        try!(write!(w, "<h3 id='{}.{}' class='method'><code>",
                      shortty(m.item()),
                      *m.item().name.get_ref()));
        try!(render_method(w, m.item()));
        try!(write!(w, "</code></h3>"));
        try!(document(w, m.item()));
        Ok(())
    }

    // Output the documentation for each function individually
    if required.len() > 0 {
        try!(write!(w, "
            <h2 id='required-methods'>Required Methods</h2>
            <div class='methods'>
        "));
        for m in required.iter() {
            try!(meth(w, *m));
        }
        try!(write!(w, "</div>"));
    }
    if provided.len() > 0 {
        try!(write!(w, "
            <h2 id='provided-methods'>Provided Methods</h2>
            <div class='methods'>
        "));
        for m in provided.iter() {
            try!(meth(w, *m));
        }
        try!(write!(w, "</div>"));
    }

    match cache_key.get().unwrap().implementors.find(&it.id) {
        Some(implementors) => {
            try!(write!(w, "
                <h2 id='implementors'>Implementors</h2>
                <ul class='item-list'>
            "));
            for i in implementors.iter() {
                match *i {
                    PathType(ref ty) => {
                        try!(write!(w, "<li><code>{}</code></li>", *ty));
                    }
                    OtherType(ref generics, ref trait_, ref for_) => {
                        try!(write!(w, "<li><code>impl{} {} for {}</code></li>",
                                      *generics, *trait_, *for_));
                    }
                }
            }
            try!(write!(w, "</ul>"));
        }
        None => {}
    }
    Ok(())
}

fn render_method(w: &mut Writer, meth: &clean::Item) -> fmt::Result {
    fn fun(w: &mut Writer, it: &clean::Item, fn_style: ast::FnStyle,
           g: &clean::Generics, selfty: &clean::SelfTy,
           d: &clean::FnDecl) -> fmt::Result {
        write!(w, "{}fn <a href='\\#{ty}.{name}' class='fnname'>{name}</a>\
                   {generics}{decl}",
               match fn_style {
                   ast::UnsafeFn => "unsafe ",
                   _ => "",
               },
               ty = shortty(it),
               name = it.name.get_ref().as_slice(),
               generics = *g,
               decl = Method(selfty, d))
    }
    match meth.inner {
        clean::TyMethodItem(ref m) => {
            fun(w, meth, m.fn_style, &m.generics, &m.self_, &m.decl)
        }
        clean::MethodItem(ref m) => {
            fun(w, meth, m.fn_style, &m.generics, &m.self_, &m.decl)
        }
        _ => unreachable!()
    }
}

fn item_struct(w: &mut Writer, it: &clean::Item,
               s: &clean::Struct) -> fmt::Result {
    try!(write!(w, "<pre class='rust struct'>"));
    try!(render_struct(w,
                       it,
                       Some(&s.generics),
                       s.struct_type,
                       s.fields.as_slice(),
                       "",
                       true));
    try!(write!(w, "</pre>"));

    try!(document(w, it));
    let mut fields = s.fields.iter().filter(|f| {
        match f.inner {
            clean::StructFieldItem(clean::HiddenStructField) => false,
            clean::StructFieldItem(clean::TypedStructField(..)) => true,
            _ => false,
        }
    }).peekable();
    match s.struct_type {
        doctree::Plain if fields.peek().is_some() => {
            try!(write!(w, "<h2 class='fields'>Fields</h2>\n<table>"));
            for field in fields {
                try!(write!(w, "<tr><td id='structfield.{name}'>\
                                  <code>{name}</code></td><td>",
                              name = field.name.get_ref().as_slice()));
                try!(document(w, field));
                try!(write!(w, "</td></tr>"));
            }
            try!(write!(w, "</table>"));
        }
        _ => {}
    }
    render_methods(w, it)
}

fn item_enum(w: &mut Writer, it: &clean::Item, e: &clean::Enum) -> fmt::Result {
    try!(write!(w, "<pre class='rust enum'>{}enum {}{}",
                  VisSpace(it.visibility),
                  it.name.get_ref().as_slice(),
                  e.generics));
    if e.variants.len() == 0 && !e.variants_stripped {
        try!(write!(w, " \\{\\}"));
    } else {
        try!(write!(w, " \\{\n"));
        for v in e.variants.iter() {
            try!(write!(w, "    "));
            let name = v.name.get_ref().as_slice();
            match v.inner {
                clean::VariantItem(ref var) => {
                    match var.kind {
                        clean::CLikeVariant => try!(write!(w, "{}", name)),
                        clean::TupleVariant(ref tys) => {
                            try!(write!(w, "{}(", name));
                            for (i, ty) in tys.iter().enumerate() {
                                if i > 0 {
                                    try!(write!(w, ", "))
                                }
                                try!(write!(w, "{}", *ty));
                            }
                            try!(write!(w, ")"));
                        }
                        clean::StructVariant(ref s) => {
                            try!(render_struct(w,
                                               v,
                                               None,
                                               s.struct_type,
                                               s.fields.as_slice(),
                                               "    ",
                                               false));
                        }
                    }
                }
                _ => unreachable!()
            }
            try!(write!(w, ",\n"));
        }

        if e.variants_stripped {
            try!(write!(w, "    // some variants omitted\n"));
        }
        try!(write!(w, "\\}"));
    }
    try!(write!(w, "</pre>"));

    try!(document(w, it));
    if e.variants.len() > 0 {
        try!(write!(w, "<h2 class='variants'>Variants</h2>\n<table>"));
        for variant in e.variants.iter() {
            try!(write!(w, "<tr><td id='variant.{name}'><code>{name}</code></td><td>",
                          name = variant.name.get_ref().as_slice()));
            try!(document(w, variant));
            match variant.inner {
                clean::VariantItem(ref var) => {
                    match var.kind {
                        clean::StructVariant(ref s) => {
                            let mut fields = s.fields.iter().filter(|f| {
                                match f.inner {
                                    clean::StructFieldItem(ref t) => match *t {
                                        clean::HiddenStructField => false,
                                        clean::TypedStructField(..) => true,
                                    },
                                    _ => false,
                                }
                            });
                            try!(write!(w, "<h3 class='fields'>Fields</h3>\n
                                              <table>"));
                            for field in fields {
                                try!(write!(w, "<tr><td \
                                                  id='variant.{v}.field.{f}'>\
                                                  <code>{f}</code></td><td>",
                                              v = variant.name.get_ref().as_slice(),
                                              f = field.name.get_ref().as_slice()));
                                try!(document(w, field));
                                try!(write!(w, "</td></tr>"));
                            }
                            try!(write!(w, "</table>"));
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            try!(write!(w, "</td></tr>"));
        }
        try!(write!(w, "</table>"));

    }
    try!(render_methods(w, it));
    Ok(())
}

fn render_struct(w: &mut Writer, it: &clean::Item,
                 g: Option<&clean::Generics>,
                 ty: doctree::StructType,
                 fields: &[clean::Item],
                 tab: &str,
                 structhead: bool) -> fmt::Result {
    try!(write!(w, "{}{}{}",
                  VisSpace(it.visibility),
                  if structhead {"struct "} else {""},
                  it.name.get_ref().as_slice()));
    match g {
        Some(g) => try!(write!(w, "{}", *g)),
        None => {}
    }
    match ty {
        doctree::Plain => {
            try!(write!(w, " \\{\n{}", tab));
            let mut fields_stripped = false;
            for field in fields.iter() {
                match field.inner {
                    clean::StructFieldItem(clean::HiddenStructField) => {
                        fields_stripped = true;
                    }
                    clean::StructFieldItem(clean::TypedStructField(ref ty)) => {
                        try!(write!(w, "    {}{}: {},\n{}",
                                      VisSpace(field.visibility),
                                      field.name.get_ref().as_slice(),
                                      *ty,
                                      tab));
                    }
                    _ => unreachable!(),
                };
            }

            if fields_stripped {
                try!(write!(w, "    // some fields omitted\n{}", tab));
            }
            try!(write!(w, "\\}"));
        }
        doctree::Tuple | doctree::Newtype => {
            try!(write!(w, "("));
            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    try!(write!(w, ", "));
                }
                match field.inner {
                    clean::StructFieldItem(clean::HiddenStructField) => {
                        try!(write!(w, "_"))
                    }
                    clean::StructFieldItem(clean::TypedStructField(ref ty)) => {
                        try!(write!(w, "{}{}", VisSpace(field.visibility), *ty))
                    }
                    _ => unreachable!()
                }
            }
            try!(write!(w, ");"));
        }
        doctree::Unit => {
            try!(write!(w, ";"));
        }
    }
    Ok(())
}

fn render_methods(w: &mut Writer, it: &clean::Item) -> fmt::Result {
    match cache_key.get().unwrap().impls.find(&it.id) {
        Some(v) => {
            let mut non_trait = v.iter().filter(|p| {
                p.ref0().trait_.is_none()
            });
            let non_trait = non_trait.collect::<Vec<&(clean::Impl, Option<~str>)>>();
            let mut traits = v.iter().filter(|p| {
                p.ref0().trait_.is_some()
            });
            let traits = traits.collect::<Vec<&(clean::Impl, Option<~str>)>>();

            if non_trait.len() > 0 {
                try!(write!(w, "<h2 id='methods'>Methods</h2>"));
                for &(ref i, ref dox) in non_trait.move_iter() {
                    try!(render_impl(w, i, dox));
                }
            }
            if traits.len() > 0 {
                try!(write!(w, "<h2 id='implementations'>Trait \
                                  Implementations</h2>"));
                let mut any_derived = false;
                for & &(ref i, ref dox) in traits.iter() {
                    if !i.derived {
                        try!(render_impl(w, i, dox));
                    } else {
                        any_derived = true;
                    }
                }
                if any_derived {
                    try!(write!(w, "<h3 id='derived_implementations'>Derived Implementations \
                                </h3>"));
                    for &(ref i, ref dox) in traits.move_iter() {
                        if i.derived {
                            try!(render_impl(w, i, dox));
                        }
                    }
                }
            }
        }
        None => {}
    }
    Ok(())
}

fn render_impl(w: &mut Writer, i: &clean::Impl,
               dox: &Option<~str>) -> fmt::Result {
    try!(write!(w, "<h3 class='impl'><code>impl{} ", i.generics));
    let trait_id = match i.trait_ {
        Some(ref ty) => {
            try!(write!(w, "{} for ", *ty));
            match *ty {
                clean::ResolvedPath { id, .. } => Some(id),
                _ => None,
            }
        }
        None => None
    };
    try!(write!(w, "{}</code></h3>", i.for_));
    match *dox {
        Some(ref dox) => {
            try!(write!(w, "<div class='docblock'>{}</div>",
                          Markdown(dox.as_slice())));
        }
        None => {}
    }

    fn docmeth(w: &mut Writer, item: &clean::Item,
               dox: bool) -> io::IoResult<()> {
        try!(write!(w, "<h4 id='method.{}' class='method'><code>",
                      *item.name.get_ref()));
        try!(render_method(w, item));
        try!(write!(w, "</code></h4>\n"));
        match item.doc_value() {
            Some(s) if dox => {
                try!(write!(w, "<div class='docblock'>{}</div>", Markdown(s)));
                Ok(())
            }
            Some(..) | None => Ok(())
        }
    }

    try!(write!(w, "<div class='methods'>"));
    for meth in i.methods.iter() {
        try!(docmeth(w, meth, true));
    }

    // If we've implemented a trait, then also emit documentation for all
    // default methods which weren't overridden in the implementation block.
    match trait_id {
        None => {}
        Some(id) => {
            try!({
                match cache_key.get().unwrap().traits.find(&id) {
                    Some(t) => {
                        for method in t.methods.iter() {
                            let n = method.item().name.clone();
                            match i.methods.iter().find(|m| m.name == n) {
                                Some(..) => continue,
                                None => {}
                            }

                            try!(docmeth(w, method.item(), false));
                        }
                    }
                    None => {}
                }
                Ok(())
            })
        }
    }
    try!(write!(w, "</div>"));
    Ok(())
}

fn item_typedef(w: &mut Writer, it: &clean::Item,
                t: &clean::Typedef) -> fmt::Result {
    try!(write!(w, "<pre class='rust typedef'>type {}{} = {};</pre>",
                  it.name.get_ref().as_slice(),
                  t.generics,
                  t.type_));

    document(w, it)
}

impl<'a> fmt::Show for Sidebar<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let cx = self.cx;
        let it = self.item;
        try!(write!(fmt.buf, "<p class='location'>"));
        let len = cx.current.len() - if it.is_mod() {1} else {0};
        for (i, name) in cx.current.iter().take(len).enumerate() {
            if i > 0 {
                try!(write!(fmt.buf, "&\\#8203;::"));
            }
            try!(write!(fmt.buf, "<a href='{}index.html'>{}</a>",
                          cx.root_path
                            .as_slice()
                            .slice_to((cx.current.len() - i - 1) * 3),
                          *name));
        }
        try!(write!(fmt.buf, "</p>"));

        fn block(w: &mut Writer, short: &str, longty: &str,
                 cur: &clean::Item, cx: &Context) -> fmt::Result {
            let items = match cx.sidebar.find_equiv(&short) {
                Some(items) => items.as_slice(),
                None => return Ok(())
            };
            try!(write!(w, "<div class='block {}'><h2>{}</h2>", short, longty));
            for item in items.iter() {
                let curty = shortty(cur).to_static_str();
                let class = if cur.name.get_ref() == item &&
                               short == curty { "current" } else { "" };
                try!(write!(w, "<a class='{ty} {class}' href='{curty, select,
                                mod{../}
                                other{}
                           }{tysel, select,
                                mod{{name}/index.html}
                                other{#.{name}.html}
                           }'>{name}</a><br/>",
                       ty = short,
                       tysel = short,
                       class = class,
                       curty = curty,
                       name = item.as_slice()));
            }
            try!(write!(w, "</div>"));
            Ok(())
        }

        try!(block(fmt.buf, "mod", "Modules", it, cx));
        try!(block(fmt.buf, "struct", "Structs", it, cx));
        try!(block(fmt.buf, "enum", "Enums", it, cx));
        try!(block(fmt.buf, "trait", "Traits", it, cx));
        try!(block(fmt.buf, "fn", "Functions", it, cx));
        Ok(())
    }
}

fn build_sidebar(m: &clean::Module) -> HashMap<~str, Vec<~str> > {
    let mut map = HashMap::new();
    for item in m.items.iter() {
        let short = shortty(item).to_static_str();
        let myname = match item.name {
            None => continue,
            Some(ref s) => s.to_owned(),
        };
        let v = map.find_or_insert_with(short.to_owned(), |_| Vec::new());
        v.push(myname);
    }

    for (_, items) in map.mut_iter() {
        items.as_mut_slice().sort();
    }
    return map;
}

impl<'a> fmt::Show for Source<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Source(s) = *self;
        let lines = s.lines().len();
        let mut cols = 0;
        let mut tmp = lines;
        while tmp > 0 {
            cols += 1;
            tmp /= 10;
        }
        try!(write!(fmt.buf, "<pre class='line-numbers'>"));
        for i in range(1, lines + 1) {
            try!(write!(fmt.buf, "<span id='{0:u}'>{0:1$u}</span>\n", i, cols));
        }
        try!(write!(fmt.buf, "</pre>"));
        try!(write!(fmt.buf, "{}", highlight::highlight(s.as_slice(), None)));
        Ok(())
    }
}

fn item_macro(w: &mut Writer, it: &clean::Item,
              t: &clean::Macro) -> fmt::Result {
    try!(w.write_str(highlight::highlight(t.source, Some("macro"))));
    document(w, it)
}
