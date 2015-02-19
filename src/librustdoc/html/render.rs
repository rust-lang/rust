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
pub use self::ExternalLocation::*;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::fmt;
use std::old_io::fs::PathExtensions;
use std::old_io::{fs, File, BufferedWriter, BufferedReader};
use std::old_io;
use std::iter::repeat;
use std::str;
use std::sync::Arc;

use externalfiles::ExternalHtml;

use serialize::json;
use serialize::json::ToJson;
use syntax::abi;
use syntax::ast;
use syntax::ast_util;
use rustc::util::nodemap::NodeSet;

use clean;
use doctree;
use fold::DocFolder;
use html::format::{VisSpace, Method, UnsafetySpace, MutableSpace, Stability};
use html::format::{ConciseStability, TyParamBounds, WhereClause};
use html::highlight;
use html::item_type::ItemType;
use html::layout;
use html::markdown::Markdown;
use html::markdown;
use html::escape::Escape;
use stability_summary;

/// A pair of name and its optional document.
#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct NameDoc(String, Option<String>);

/// Major driving force in all rustdoc rendering. This contains information
/// about where in the tree-like hierarchy rendering is occurring and controls
/// how the current page is being rendered.
///
/// It is intended that this context is a lightweight object which can be fairly
/// easily cloned because it is cloned per work-job (about once per item in the
/// rustdoc tree).
#[derive(Clone)]
pub struct Context {
    /// Current hierarchy of components leading down to what's currently being
    /// rendered
    pub current: Vec<String>,
    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    pub root_path: String,
    /// The path to the crate root source minus the file name.
    /// Used for simplifying paths to the highlighted source code files.
    pub src_root: Path,
    /// The current destination folder of where HTML artifacts should be placed.
    /// This changes as the context descends into the module hierarchy.
    pub dst: Path,
    /// This describes the layout of each page, and is not modified after
    /// creation of the context (contains info like the favicon and added html).
    pub layout: layout::Layout,
    /// This map is a list of what should be displayed on the sidebar of the
    /// current page. The key is the section header (traits, modules,
    /// functions), and the value is the list of containers belonging to this
    /// header. This map will change depending on the surrounding context of the
    /// page.
    pub sidebar: HashMap<String, Vec<NameDoc>>,
    /// This flag indicates whether [src] links should be generated or not. If
    /// the source files are present in the html rendering, then this will be
    /// `true`.
    pub include_sources: bool,
    /// A flag, which when turned off, will render pages which redirect to the
    /// real location of an item. This is used to allow external links to
    /// publicly reused items to redirect to the right location.
    pub render_redirect_pages: bool,
    /// All the passes that were run on this crate.
    pub passes: HashSet<String>,
}

/// Indicates where an external crate can be found.
pub enum ExternalLocation {
    /// Remote URL root of the external crate
    Remote(String),
    /// This external crate can be found in the local doc/ folder
    Local,
    /// The external crate could not be found.
    Unknown,
}

/// Metadata about an implementor of a trait.
pub struct Implementor {
    pub def_id: ast::DefId,
    pub generics: clean::Generics,
    pub trait_: clean::Type,
    pub for_: clean::Type,
    pub stability: Option<clean::Stability>,
}

/// Metadata about implementations for a type.
#[derive(Clone)]
pub struct Impl {
    pub impl_: clean::Impl,
    pub dox: Option<String>,
    pub stability: Option<clean::Stability>,
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
#[derive(Default)]
pub struct Cache {
    /// Mapping of typaram ids to the name of the type parameter. This is used
    /// when pretty-printing a type (so pretty printing doesn't have to
    /// painfully maintain a context like this)
    pub typarams: HashMap<ast::DefId, String>,

    /// Maps a type id to all known implementations for that type. This is only
    /// recognized for intra-crate `ResolvedPath` types, and is used to print
    /// out extra documentation on the page of an enum/struct.
    ///
    /// The values of the map are a list of implementations and documentation
    /// found on that implementation.
    pub impls: HashMap<ast::DefId, Vec<Impl>>,

    /// Maintains a mapping of local crate node ids to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    pub paths: HashMap<ast::DefId, (Vec<String>, ItemType)>,

    /// Similar to `paths`, but only holds external paths. This is only used for
    /// generating explicit hyperlinks to other crates.
    pub external_paths: HashMap<ast::DefId, Vec<String>>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    pub traits: HashMap<ast::DefId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    pub implementors: HashMap<ast::DefId, Vec<Implementor>>,

    /// Cache of where external crate documentation can be found.
    pub extern_locations: HashMap<ast::CrateNum, ExternalLocation>,

    /// Cache of where documentation for primitives can be found.
    pub primitive_locations: HashMap<clean::PrimitiveType, ast::CrateNum>,

    /// Set of definitions which have been inlined from external crates.
    pub inlined: HashSet<ast::DefId>,

    // Private fields only used when initially crawling a crate to build a cache

    stack: Vec<String>,
    parent_stack: Vec<ast::DefId>,
    search_index: Vec<IndexItem>,
    privmod: bool,
    remove_priv: bool,
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
    seen: HashSet<String>,
    /// Root destination to place all HTML output into
    dst: Path,
}

/// Wrapper struct to render the source code of a file. This will do things like
/// adding line numbers to the left-hand side.
struct Source<'a>(&'a str);

// Helper structs for rendering items/sidebars and carrying along contextual
// information

#[derive(Copy)]
struct Item<'a> {
    cx: &'a Context,
    item: &'a clean::Item,
}

struct Sidebar<'a> { cx: &'a Context, item: &'a clean::Item, }

/// Struct representing one entry in the JS search index. These are all emitted
/// by hand to a large JS file at the end of cache-creation.
struct IndexItem {
    ty: ItemType,
    name: String,
    path: String,
    desc: String,
    parent: Option<ast::DefId>,
}

// TLS keys used to carry information around during rendering.

thread_local!(static CACHE_KEY: RefCell<Arc<Cache>> = Default::default());
thread_local!(pub static CURRENT_LOCATION_KEY: RefCell<Vec<String>> =
                    RefCell::new(Vec::new()));

/// Generates the documentation for `crate` into the directory `dst`
pub fn run(mut krate: clean::Crate,
           external_html: &ExternalHtml,
           dst: Path,
           passes: HashSet<String>) -> old_io::IoResult<()> {
    let mut cx = Context {
        dst: dst,
        src_root: krate.src.dir_path(),
        passes: passes,
        current: Vec::new(),
        root_path: String::new(),
        sidebar: HashMap::new(),
        layout: layout::Layout {
            logo: "".to_string(),
            favicon: "".to_string(),
            external_html: external_html.clone(),
            krate: krate.name.clone(),
            playground_url: "".to_string(),
        },
        include_sources: true,
        render_redirect_pages: false,
    };

    try!(mkdir(&cx.dst));

    // Crawl the crate attributes looking for attributes which control how we're
    // going to emit HTML
    let default: &[_] = &[];
    match krate.module.as_ref().map(|m| m.doc_list().unwrap_or(default)) {
        Some(attrs) => {
            for attr in attrs {
                match *attr {
                    clean::NameValue(ref x, ref s)
                            if "html_favicon_url" == *x => {
                        cx.layout.favicon = s.to_string();
                    }
                    clean::NameValue(ref x, ref s)
                            if "html_logo_url" == *x => {
                        cx.layout.logo = s.to_string();
                    }
                    clean::NameValue(ref x, ref s)
                            if "html_playground_url" == *x => {
                        cx.layout.playground_url = s.to_string();
                        markdown::PLAYGROUND_KRATE.with(|slot| {
                            if slot.borrow().is_none() {
                                let name = krate.name.clone();
                                *slot.borrow_mut() = Some(Some(name));
                            }
                        });
                    }
                    clean::Word(ref x)
                            if "html_no_source" == *x => {
                        cx.include_sources = false;
                    }
                    _ => {}
                }
            }
        }
        None => {}
    }

    // Crawl the crate to build various caches used for the output
    let analysis = ::ANALYSISKEY.with(|a| a.clone());
    let analysis = analysis.borrow();
    let public_items = analysis.as_ref().map(|a| a.public_items.clone());
    let public_items = public_items.unwrap_or(NodeSet());
    let paths: HashMap<ast::DefId, (Vec<String>, ItemType)> =
      analysis.as_ref().map(|a| {
        let paths = a.external_paths.borrow_mut().take().unwrap();
        paths.into_iter().map(|(k, (v, t))| (k, (v, ItemType::from_type_kind(t)))).collect()
      }).unwrap_or(HashMap::new());
    let mut cache = Cache {
        impls: HashMap::new(),
        external_paths: paths.iter().map(|(&k, v)| (k, v.0.clone()))
                             .collect(),
        paths: paths,
        implementors: HashMap::new(),
        stack: Vec::new(),
        parent_stack: Vec::new(),
        search_index: Vec::new(),
        extern_locations: HashMap::new(),
        primitive_locations: HashMap::new(),
        remove_priv: cx.passes.contains("strip-private"),
        privmod: false,
        public_items: public_items,
        orphan_methods: Vec::new(),
        traits: analysis.as_ref().map(|a| {
            a.external_traits.borrow_mut().take().unwrap()
        }).unwrap_or(HashMap::new()),
        typarams: analysis.as_ref().map(|a| {
            a.external_typarams.borrow_mut().take().unwrap()
        }).unwrap_or(HashMap::new()),
        inlined: analysis.as_ref().map(|a| {
            a.inlined.borrow_mut().take().unwrap()
        }).unwrap_or(HashSet::new()),
    };
    cache.stack.push(krate.name.clone());
    krate = cache.fold_crate(krate);

    // Cache where all our extern crates are located
    for &(n, ref e) in &krate.externs {
        cache.extern_locations.insert(n, extern_location(e, &cx.dst));
        let did = ast::DefId { krate: n, node: ast::CRATE_NODE_ID };
        cache.paths.insert(did, (vec![e.name.to_string()], ItemType::Module));
    }

    // Cache where all known primitives have their documentation located.
    //
    // Favor linking to as local extern as possible, so iterate all crates in
    // reverse topological order.
    for &(n, ref e) in krate.externs.iter().rev() {
        for &prim in &e.primitives {
            cache.primitive_locations.insert(prim, n);
        }
    }
    for &prim in &krate.primitives {
        cache.primitive_locations.insert(prim, ast::LOCAL_CRATE);
    }

    // Build our search index
    let index = try!(build_index(&krate, &mut cache));

    // Freeze the cache now that the index has been built. Put an Arc into TLS
    // for future parallelization opportunities
    let cache = Arc::new(cache);
    CACHE_KEY.with(|v| *v.borrow_mut() = cache.clone());
    CURRENT_LOCATION_KEY.with(|s| s.borrow_mut().clear());

    try!(write_shared(&cx, &krate, &*cache, index));
    let krate = try!(render_sources(&mut cx, krate));

    // Crawl the crate, building a summary of the stability levels.
    let summary = stability_summary::build(&krate);

    // And finally render the whole crate's documentation
    cx.krate(krate, summary)
}

fn build_index(krate: &clean::Crate, cache: &mut Cache) -> old_io::IoResult<String> {
    // Build the search index from the collected metadata
    let mut nodeid_to_pathid = HashMap::new();
    let mut pathid_to_nodeid = Vec::new();
    {
        let Cache { ref mut search_index,
                    ref orphan_methods,
                    ref mut paths, .. } = *cache;

        // Attach all orphan methods to the type's definition if the type
        // has since been learned.
        for &(pid, ref item) in orphan_methods {
            let did = ast_util::local_def(pid);
            match paths.get(&did) {
                Some(&(ref fqp, _)) => {
                    search_index.push(IndexItem {
                        ty: shortty(item),
                        name: item.name.clone().unwrap(),
                        path: fqp[..fqp.len() - 1].connect("::"),
                        desc: shorter(item.doc_value()).to_string(),
                        parent: Some(did),
                    });
                },
                None => {}
            }
        };

        // Reduce `NodeId` in paths into smaller sequential numbers,
        // and prune the paths that do not appear in the index.
        for item in &*search_index {
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

    // Collect the index into a string
    let mut w = Vec::new();
    try!(write!(&mut w, r#"searchIndex['{}'] = {{"items":["#, krate.name));

    let mut lastpath = "".to_string();
    for (i, item) in cache.search_index.iter().enumerate() {
        // Omit the path if it is same to that of the prior item.
        let path;
        if lastpath == item.path {
            path = "";
        } else {
            lastpath = item.path.to_string();
            path = &item.path;
        };

        if i > 0 {
            try!(write!(&mut w, ","));
        }
        try!(write!(&mut w, r#"[{},"{}","{}",{}"#,
                    item.ty as uint, item.name, path,
                    item.desc.to_json().to_string()));
        match item.parent {
            Some(nodeid) => {
                let pathid = *nodeid_to_pathid.get(&nodeid).unwrap();
                try!(write!(&mut w, ",{}", pathid));
            }
            None => {}
        }
        try!(write!(&mut w, "]"));
    }

    try!(write!(&mut w, r#"],"paths":["#));

    for (i, &did) in pathid_to_nodeid.iter().enumerate() {
        let &(ref fqp, short) = cache.paths.get(&did).unwrap();
        if i > 0 {
            try!(write!(&mut w, ","));
        }
        try!(write!(&mut w, r#"[{},"{}"]"#,
                    short as uint, *fqp.last().unwrap()));
    }

    try!(write!(&mut w, "]}};"));

    Ok(String::from_utf8(w).unwrap())
}

fn write_shared(cx: &Context,
                krate: &clean::Crate,
                cache: &Cache,
                search_index: String) -> old_io::IoResult<()> {
    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    try!(mkdir(&cx.dst));
    let _lock = ::flock::Lock::new(&cx.dst.join(".lock"));

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.
    try!(write(cx.dst.join("jquery.js"),
               include_bytes!("static/jquery-2.1.0.min.js")));
    try!(write(cx.dst.join("main.js"), include_bytes!("static/main.js")));
    try!(write(cx.dst.join("playpen.js"), include_bytes!("static/playpen.js")));
    try!(write(cx.dst.join("main.css"), include_bytes!("static/main.css")));
    try!(write(cx.dst.join("normalize.css"),
               include_bytes!("static/normalize.css")));
    try!(write(cx.dst.join("FiraSans-Regular.woff"),
               include_bytes!("static/FiraSans-Regular.woff")));
    try!(write(cx.dst.join("FiraSans-Medium.woff"),
               include_bytes!("static/FiraSans-Medium.woff")));
    try!(write(cx.dst.join("Heuristica-Italic.woff"),
               include_bytes!("static/Heuristica-Italic.woff")));
    try!(write(cx.dst.join("SourceSerifPro-Regular.woff"),
               include_bytes!("static/SourceSerifPro-Regular.woff")));
    try!(write(cx.dst.join("SourceSerifPro-Bold.woff"),
               include_bytes!("static/SourceSerifPro-Bold.woff")));
    try!(write(cx.dst.join("SourceCodePro-Regular.woff"),
               include_bytes!("static/SourceCodePro-Regular.woff")));
    try!(write(cx.dst.join("SourceCodePro-Semibold.woff"),
               include_bytes!("static/SourceCodePro-Semibold.woff")));

    fn collect(path: &Path, krate: &str,
               key: &str) -> old_io::IoResult<Vec<String>> {
        let mut ret = Vec::new();
        if path.exists() {
            for line in BufferedReader::new(File::open(path)).lines() {
                let line = try!(line);
                if !line.starts_with(key) {
                    continue
                }
                if line.starts_with(&format!("{}['{}']", key, krate)) {
                    continue
                }
                ret.push(line.to_string());
            }
        }
        return Ok(ret);
    }

    // Update the search index
    let dst = cx.dst.join("search-index.js");
    let all_indexes = try!(collect(&dst, &krate.name, "searchIndex"));
    let mut w = try!(File::create(&dst));
    try!(writeln!(&mut w, "var searchIndex = {{}};"));
    try!(writeln!(&mut w, "{}", search_index));
    for index in &all_indexes {
        try!(writeln!(&mut w, "{}", *index));
    }
    try!(writeln!(&mut w, "initSearch(searchIndex);"));

    // Update the list of all implementors for traits
    let dst = cx.dst.join("implementors");
    try!(mkdir(&dst));
    for (&did, imps) in &cache.implementors {
        // Private modules can leak through to this phase of rustdoc, which
        // could contain implementations for otherwise private types. In some
        // rare cases we could find an implementation for an item which wasn't
        // indexed, so we just skip this step in that case.
        //
        // FIXME: this is a vague explanation for why this can't be a `get`, in
        //        theory it should be...
        let &(ref remote_path, remote_item_type) = match cache.paths.get(&did) {
            Some(p) => p,
            None => continue,
        };

        let mut mydst = dst.clone();
        for part in &remote_path[..remote_path.len() - 1] {
            mydst.push(part);
            try!(mkdir(&mydst));
        }
        mydst.push(format!("{}.{}.js",
                           remote_item_type.to_static_str(),
                           remote_path[remote_path.len() - 1]));
        let all_implementors = try!(collect(&mydst, &krate.name,
                                            "implementors"));

        try!(mkdir(&mydst.dir_path()));
        let mut f = BufferedWriter::new(try!(File::create(&mydst)));
        try!(writeln!(&mut f, "(function() {{var implementors = {{}};"));

        for implementor in &all_implementors {
            try!(write!(&mut f, "{}", *implementor));
        }

        try!(write!(&mut f, r"implementors['{}'] = [", krate.name));
        for imp in imps {
            // If the trait and implementation are in the same crate, then
            // there's no need to emit information about it (there's inlining
            // going on). If they're in different crates then the crate defining
            // the trait will be interested in our implementation.
            if imp.def_id.krate == did.krate { continue }
            try!(write!(&mut f, r#""{}impl{} {} for {}","#,
                        ConciseStability(&imp.stability),
                        imp.generics, imp.trait_, imp.for_));
        }
        try!(writeln!(&mut f, r"];"));
        try!(writeln!(&mut f, "{}", r"
            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        "));
        try!(writeln!(&mut f, r"}})()"));
    }
    Ok(())
}

fn render_sources(cx: &mut Context,
                  krate: clean::Crate) -> old_io::IoResult<clean::Crate> {
    info!("emitting source files");
    let dst = cx.dst.join("src");
    try!(mkdir(&dst));
    let dst = dst.join(&krate.name);
    try!(mkdir(&dst));
    let mut folder = SourceCollector {
        dst: dst,
        seen: HashSet::new(),
        cx: cx,
    };
    // skip all invalid spans
    folder.seen.insert("".to_string());
    Ok(folder.fold_crate(krate))
}

/// Writes the entire contents of a string to a destination, not attempting to
/// catch any errors.
fn write(dst: Path, contents: &[u8]) -> old_io::IoResult<()> {
    File::create(&dst).write_all(contents)
}

/// Makes a directory on the filesystem, failing the task if an error occurs and
/// skipping if the directory already exists.
fn mkdir(path: &Path) -> old_io::IoResult<()> {
    if !path.exists() {
        fs::mkdir(path, old_io::USER_RWX)
    } else {
        Ok(())
    }
}

/// Returns a documentation-level item type from the item.
fn shortty(item: &clean::Item) -> ItemType {
    ItemType::from_item(item)
}

/// Takes a path to a source file and cleans the path to it. This canonicalizes
/// things like ".." to components which preserve the "top down" hierarchy of a
/// static HTML tree.
// FIXME (#9639): The closure should deal with &[u8] instead of &str
// FIXME (#9639): This is too conservative, rejecting non-UTF-8 paths
fn clean_srcpath<F>(src_root: &Path, src: &[u8], mut f: F) where
    F: FnMut(&str),
{
    let p = Path::new(src);

    // make it relative, if possible
    let p = p.path_relative_from(src_root).unwrap_or(p);

    if p.as_vec() != b"." {
        for c in p.str_components().map(|x|x.unwrap()) {
            if ".." == c {
                f("up");
            } else {
                f(c)
            }
        }
    }
}

/// Attempts to find where an external crate is located, given that we're
/// rendering in to the specified source destination.
fn extern_location(e: &clean::ExternalCrate, dst: &Path) -> ExternalLocation {
    // See if there's documentation generated into the local directory
    let local_location = dst.join(&e.name);
    if local_location.is_dir() {
        return Local;
    }

    // Failing that, see if there's an attribute specifying where to find this
    // external crate
    for attr in &e.attrs {
        match *attr {
            clean::List(ref x, ref list) if "doc" == *x => {
                for attr in list {
                    match *attr {
                        clean::NameValue(ref x, ref s)
                                if "html_root_url" == *x => {
                            if s.ends_with("/") {
                                return Remote(s.to_string());
                            }
                            return Remote(format!("{}/", s));
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
            self.cx
                .include_sources = match self.emit_source(&item.source .filename) {
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
    fn emit_source(&mut self, filename: &str) -> old_io::IoResult<()> {
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
        let contents = str::from_utf8(&contents).unwrap();

        // Remove the utf-8 BOM if any
        let contents = if contents.starts_with("\u{feff}") {
            &contents[3..]
        } else {
            contents
        };

        // Create the intermediate directories
        let mut cur = self.dst.clone();
        let mut root_path = String::from_str("../../");
        clean_srcpath(&self.cx.src_root, p.dirname(), |component| {
            cur.push(component);
            mkdir(&cur).unwrap();
            root_path.push_str("../");
        });

        let mut fname = p.filename().expect("source has no filename").to_vec();
        fname.extend(".html".bytes());
        cur.push(fname);
        let mut w = BufferedWriter::new(try!(File::create(&cur)));

        let title = format!("{} -- source", cur.filename_display());
        let desc = format!("Source to the Rust file `{}`.", filename);
        let page = layout::Page {
            title: &title,
            ty: "source",
            root_path: &root_path,
            description: &desc,
            keywords: get_basic_keywords(),
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
                self.privmod = prev || (self.remove_priv && item.visibility != Some(ast::Public));
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
        if let clean::TraitItem(ref t) = item.inner {
            self.traits.insert(item.def_id, t.clone());
        }

        // Collect all the implementors of traits.
        if let clean::ImplItem(ref i) = item.inner {
            match i.trait_ {
                Some(clean::ResolvedPath{ did, .. }) => {
                    let v = self.implementors.entry(did).get().unwrap_or_else(
                        |vacant_entry| vacant_entry.insert(Vec::with_capacity(1)));
                    v.push(Implementor {
                        def_id: item.def_id,
                        generics: i.generics.clone(),
                        trait_: i.trait_.as_ref().unwrap().clone(),
                        for_: i.for_.clone(),
                        stability: item.stability.clone(),
                    });
                }
                Some(..) | None => {}
            }
        }

        // Index this method for searching later on
        if let Some(ref s) = item.name {
            let (parent, is_method) = match item.inner {
                clean::TyMethodItem(..) |
                clean::StructFieldItem(..) |
                clean::VariantItem(..) => {
                    ((Some(*self.parent_stack.last().unwrap()),
                      Some(&self.stack[..self.stack.len() - 1])),
                     false)
                }
                clean::MethodItem(..) => {
                    if self.parent_stack.len() == 0 {
                        ((None, None), false)
                    } else {
                        let last = self.parent_stack.last().unwrap();
                        let did = *last;
                        let path = match self.paths.get(&did) {
                            Some(&(_, ItemType::Trait)) =>
                                Some(&self.stack[..self.stack.len() - 1]),
                            // The current stack not necessarily has correlation for
                            // where the type was defined. On the other hand,
                            // `paths` always has the right information if present.
                            Some(&(ref fqp, ItemType::Struct)) |
                            Some(&(ref fqp, ItemType::Enum)) =>
                                Some(&fqp[..fqp.len() - 1]),
                            Some(..) => Some(&*self.stack),
                            None => None
                        };
                        ((Some(*last), path), true)
                    }
                }
                _ => ((None, Some(&*self.stack)), false)
            };
            let hidden_field = match item.inner {
                clean::StructFieldItem(clean::HiddenStructField) => true,
                _ => false
            };

            match parent {
                (parent, Some(path)) if is_method || (!self.privmod && !hidden_field) => {
                    self.search_index.push(IndexItem {
                        ty: shortty(&item),
                        name: s.to_string(),
                        path: path.connect("::").to_string(),
                        desc: shorter(item.doc_value()).to_string(),
                        parent: parent,
                    });
                }
                (Some(parent), None) if is_method || (!self.privmod && !hidden_field)=> {
                    if ast_util::is_local(parent) {
                        // We have a parent, but we don't know where they're
                        // defined yet. Wait for later to index this item.
                        self.orphan_methods.push((parent.node, item.clone()))
                    }
                }
                _ => {}
            }
        }

        // Keep track of the fully qualified path for this item.
        let pushed = if item.name.is_some() {
            let n = item.name.as_ref().unwrap();
            if n.len() > 0 {
                self.stack.push(n.to_string());
                true
            } else { false }
        } else { false };
        match item.inner {
            clean::StructItem(..) | clean::EnumItem(..) |
            clean::TypedefItem(..) | clean::TraitItem(..) |
            clean::FunctionItem(..) | clean::ModuleItem(..) |
            clean::ForeignFunctionItem(..) if !self.privmod => {
                // Reexported items mean that the same id can show up twice
                // in the rustdoc ast that we're looking at. We know,
                // however, that a reexported item doesn't show up in the
                // `public_items` map, so we can skip inserting into the
                // paths map if there was already an entry present and we're
                // not a public item.
                let id = item.def_id.node;
                if !self.paths.contains_key(&item.def_id) ||
                   !ast_util::is_local(item.def_id) ||
                   self.public_items.contains(&id) {
                    self.paths.insert(item.def_id,
                                      (self.stack.clone(), shortty(&item)));
                }
            }
            // link variants to their parent enum because pages aren't emitted
            // for each variant
            clean::VariantItem(..) if !self.privmod => {
                let mut stack = self.stack.clone();
                stack.pop();
                self.paths.insert(item.def_id, (stack, ItemType::Enum));
            }

            clean::PrimitiveItem(..) if item.visibility.is_some() => {
                self.paths.insert(item.def_id, (self.stack.clone(),
                                                shortty(&item)));
            }

            _ => {}
        }

        // Maintain the parent stack
        let parent_pushed = match item.inner {
            clean::TraitItem(..) | clean::EnumItem(..) | clean::StructItem(..) => {
                self.parent_stack.push(item.def_id);
                true
            }
            clean::ImplItem(ref i) => {
                match i.for_ {
                    clean::ResolvedPath{ did, .. } => {
                        self.parent_stack.push(did);
                        true
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
                        use clean::{Primitive, Vector, ResolvedPath, BorrowedRef};
                        use clean::{FixedVector, Slice, Tuple, PrimitiveTuple};

                        // extract relevant documentation for this impl
                        let dox = match attrs.into_iter().find(|a| {
                            match *a {
                                clean::NameValue(ref x, _)
                                        if "doc" == *x => {
                                    true
                                }
                                _ => false
                            }
                        }) {
                            Some(clean::NameValue(_, dox)) => Some(dox),
                            Some(..) | None => None,
                        };

                        // Figure out the id of this impl. This may map to a
                        // primitive rather than always to a struct/enum.
                        let did = match i.for_ {
                            ResolvedPath { did, .. } => Some(did),

                            // References to primitives are picked up as well to
                            // recognize implementations for &str, this may not
                            // be necessary in a DST world.
                            Primitive(p) |
                                BorrowedRef { type_: box Primitive(p), ..} =>
                            {
                                Some(ast_util::local_def(p.to_node_id()))
                            }

                            // In a DST world, we may only need
                            // Vector/FixedVector, but for now we also pick up
                            // borrowed references
                            Vector(..) | FixedVector(..) |
                                BorrowedRef{ type_: box Vector(..), ..  } |
                                BorrowedRef{ type_: box FixedVector(..), .. } =>
                            {
                                Some(ast_util::local_def(Slice.to_node_id()))
                            }

                            Tuple(..) => {
                                let id = PrimitiveTuple.to_node_id();
                                Some(ast_util::local_def(id))
                            }

                            _ => None,
                        };

                        if let Some(did) = did {
                            let v = self.impls.entry(did).get().unwrap_or_else(
                                |vacant_entry| vacant_entry.insert(Vec::with_capacity(1)));
                            v.push(Impl {
                                impl_: i,
                                dox: dox,
                                stability: item.stability.clone(),
                            });
                        }

                        None
                    }

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
        for typ in &generics.type_params {
            self.typarams.insert(typ.did, typ.name.clone());
        }
    }
}

impl Context {
    /// Recurse in the directory structure and change the "root path" to make
    /// sure it always points to the top (relatively)
    fn recurse<T, F>(&mut self, s: String, f: F) -> T where
        F: FnOnce(&mut Context) -> T,
    {
        if s.len() == 0 {
            panic!("Unexpected empty destination: {:?}", self.current);
        }
        let prev = self.dst.clone();
        self.dst.push(&s);
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
    fn krate(mut self, mut krate: clean::Crate,
             stability: stability_summary::ModuleSummary) -> old_io::IoResult<()> {
        let mut item = match krate.module.take() {
            Some(i) => i,
            None => return Ok(())
        };
        item.name = Some(krate.name);

        // render stability dashboard
        try!(self.recurse(stability.name.clone(), |this| {
            let json_dst = &this.dst.join("stability.json");
            let mut json_out = BufferedWriter::new(try!(File::create(json_dst)));
            try!(write!(&mut json_out, "{}", json::as_json(&stability)));

            let mut title = stability.name.clone();
            title.push_str(" - Stability dashboard");
            let desc = format!("API stability overview for the Rust `{}` crate.",
                               this.layout.krate);
            let page = layout::Page {
                ty: "mod",
                root_path: &this.root_path,
                title: &title,
                description: &desc,
                keywords: get_basic_keywords(),
            };
            let html_dst = &this.dst.join("stability.html");
            let mut html_out = BufferedWriter::new(try!(File::create(html_dst)));
            layout::render(&mut html_out, &this.layout, &page,
                           &Sidebar{ cx: this, item: &item },
                           &stability)
        }));

        // render the crate documentation
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

    /// Non-parallelized version of rendering an item. This will take the input
    /// item, render its contents, and then invoke the specified closure with
    /// all sub-items which need to be rendered.
    ///
    /// The rendering driver uses this closure to queue up more work.
    fn item<F>(&mut self, item: clean::Item, mut f: F) -> old_io::IoResult<()> where
        F: FnMut(&mut Context, clean::Item),
    {
        fn render(w: old_io::File, cx: &Context, it: &clean::Item,
                  pushname: bool) -> old_io::IoResult<()> {
            info!("Rendering an item to {}", w.path().display());
            // A little unfortunate that this is done like this, but it sure
            // does make formatting *a lot* nicer.
            CURRENT_LOCATION_KEY.with(|slot| {
                *slot.borrow_mut() = cx.current.clone();
            });

            let mut title = cx.current.connect("::");
            if pushname {
                if title.len() > 0 {
                    title.push_str("::");
                }
                title.push_str(it.name.as_ref().unwrap());
            }
            title.push_str(" - Rust");
            let tyname = shortty(it).to_static_str();
            let is_crate = match it.inner {
                clean::ModuleItem(clean::Module { items: _, is_crate: true }) => true,
                _ => false
            };
            let desc = if is_crate {
                format!("API documentation for the Rust `{}` crate.",
                        cx.layout.krate)
            } else {
                format!("API documentation for the Rust `{}` {} in crate `{}`.",
                        it.name.as_ref().unwrap(), tyname, cx.layout.krate)
            };
            let keywords = make_item_keywords(it);
            let page = layout::Page {
                ty: tyname,
                root_path: &cx.root_path,
                title: &title,
                description: &desc,
                keywords: &keywords,
            };

            markdown::reset_headers();

            // We have a huge number of calls to write, so try to alleviate some
            // of the pain by using a buffered writer instead of invoking the
            // write syscall all the time.
            let mut writer = BufferedWriter::new(w);
            if !cx.render_redirect_pages {
                try!(layout::render(&mut writer, &cx.layout, &page,
                                    &Sidebar{ cx: cx, item: it },
                                    &Item{ cx: cx, item: it }));
            } else {
                let mut url = repeat("../").take(cx.current.len())
                                           .collect::<String>();
                match cache().paths.get(&it.def_id) {
                    Some(&(ref names, _)) => {
                        for name in &names[..names.len() - 1] {
                            url.push_str(name);
                            url.push_str("/");
                        }
                        url.push_str(&item_path(it));
                        try!(layout::redirect(&mut writer, &url));
                    }
                    None => {}
                }
            }
            writer.flush()
        }

        // Private modules may survive the strip-private pass if they
        // contain impls for public types. These modules can also
        // contain items such as publicly reexported structures.
        //
        // External crates will provide links to these structures, so
        // these modules are recursed into, but not rendered normally (a
        // flag on the context).
        if !self.render_redirect_pages {
            self.render_redirect_pages = self.ignore_private_item(&item);
        }

        match item.inner {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            clean::ModuleItem(..) => {
                let name = item.name.as_ref().unwrap().to_string();
                let mut item = Some(item);
                self.recurse(name, |this| {
                    let item = item.take().unwrap();
                    let dst = this.dst.join("index.html");
                    let dst = try!(File::create(&dst));
                    try!(render(dst, this, &item, false));

                    let m = match item.inner {
                        clean::ModuleItem(m) => m,
                        _ => unreachable!()
                    };
                    this.sidebar = this.build_sidebar(&m);
                    for item in m.items {
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

    fn build_sidebar(&self, m: &clean::Module) -> HashMap<String, Vec<NameDoc>> {
        let mut map = HashMap::new();
        for item in &m.items {
            if self.ignore_private_item(item) { continue }

            // avoid putting foreign items to the sidebar.
            if let &clean::ForeignFunctionItem(..) = &item.inner { continue }
            if let &clean::ForeignStaticItem(..) = &item.inner { continue }

            let short = shortty(item).to_static_str();
            let myname = match item.name {
                None => continue,
                Some(ref s) => s.to_string(),
            };
            let short = short.to_string();
            let v = map.entry(short).get().unwrap_or_else(
                |vacant_entry| vacant_entry.insert(Vec::with_capacity(1)));
            v.push(NameDoc(myname, Some(shorter_line(item.doc_value()))));
        }

        for (_, items) in &mut map {
            items.sort();
        }
        return map;
    }

    fn ignore_private_item(&self, it: &clean::Item) -> bool {
        match it.inner {
            clean::ModuleItem(ref m) => {
                (m.items.len() == 0 && it.doc_value().is_none()) ||
                (self.passes.contains("strip-private") && it.visibility != Some(ast::Public))
            }
            clean::PrimitiveItem(..) => it.visibility != Some(ast::Public),
            _ => false,
        }
    }
}

impl<'a> Item<'a> {
    fn ismodule(&self) -> bool {
        match self.item.inner {
            clean::ModuleItem(..) => true, _ => false
        }
    }

    /// Generate a url appropriate for an `href` attribute back to the source of
    /// this item.
    ///
    /// The url generated, when clicked, will redirect the browser back to the
    /// original source code.
    ///
    /// If `None` is returned, then a source link couldn't be generated. This
    /// may happen, for example, with externally inlined items where the source
    /// of their crate documentation isn't known.
    fn href(&self, cx: &Context) -> Option<String> {
        // If this item is part of the local crate, then we're guaranteed to
        // know the span, so we plow forward and generate a proper url. The url
        // has anchors for the line numbers that we're linking to.
        if ast_util::is_local(self.item.def_id) {
            let mut path = Vec::new();
            clean_srcpath(&cx.src_root, self.item.source.filename.as_bytes(),
                          |component| {
                path.push(component.to_string());
            });
            let href = if self.item.source.loline == self.item.source.hiline {
                format!("{}", self.item.source.loline)
            } else {
                format!("{}-{}",
                        self.item.source.loline,
                        self.item.source.hiline)
            };
            Some(format!("{root}src/{krate}/{path}.html#{href}",
                         root = self.cx.root_path,
                         krate = self.cx.layout.krate,
                         path = path.connect("/"),
                         href = href))

        // If this item is not part of the local crate, then things get a little
        // trickier. We don't actually know the span of the external item, but
        // we know that the documentation on the other end knows the span!
        //
        // In this case, we generate a link to the *documentation* for this type
        // in the original crate. There's an extra URL parameter which says that
        // we want to go somewhere else, and the JS on the destination page will
        // pick it up and instantly redirect the browser to the source code.
        //
        // If we don't know where the external documentation for this crate is
        // located, then we return `None`.
        } else {
            let cache = cache();
            let path = &cache.external_paths[self.item.def_id];
            let root = match cache.extern_locations[self.item.def_id.krate] {
                Remote(ref s) => s.to_string(),
                Local => self.cx.root_path.clone(),
                Unknown => return None,
            };
            Some(format!("{root}{path}/{file}?gotosrc={goto}",
                         root = root,
                         path = path[..path.len() - 1].connect("/"),
                         file = item_path(self.item),
                         goto = self.item.def_id.node))
        }
    }
}


impl<'a> fmt::Display for Item<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        // Write the breadcrumb trail header for the top
        try!(write!(fmt, "\n<h1 class='fqn'><span class='in-band'>"));
        match self.item.inner {
            clean::ModuleItem(ref m) => if m.is_crate {
                    try!(write!(fmt, "Crate "));
                } else {
                    try!(write!(fmt, "Module "));
                },
            clean::FunctionItem(..) => try!(write!(fmt, "Function ")),
            clean::TraitItem(..) => try!(write!(fmt, "Trait ")),
            clean::StructItem(..) => try!(write!(fmt, "Struct ")),
            clean::EnumItem(..) => try!(write!(fmt, "Enum ")),
            clean::PrimitiveItem(..) => try!(write!(fmt, "Primitive Type ")),
            _ => {}
        }
        let is_primitive = match self.item.inner {
            clean::PrimitiveItem(..) => true,
            _ => false,
        };
        if !is_primitive {
            let cur = &self.cx.current;
            let amt = if self.ismodule() { cur.len() - 1 } else { cur.len() };
            for (i, component) in cur.iter().enumerate().take(amt) {
                try!(write!(fmt, "<a href='{}index.html'>{}</a>::<wbr>",
                            repeat("../").take(cur.len() - i - 1)
                                         .collect::<String>(),
                            component));
            }
        }
        try!(write!(fmt, "<a class='{}' href=''>{}</a>",
                    shortty(self.item), self.item.name.as_ref().unwrap()));

        // Write stability level
        try!(write!(fmt, "<wbr>{}", Stability(&self.item.stability)));

        try!(write!(fmt, "</span>")); // in-band
        // Links to out-of-band information, i.e. src and stability dashboard
        try!(write!(fmt, "<span class='out-of-band'>"));

        // Write stability dashboard link
        match self.item.inner {
            clean::ModuleItem(ref m) if m.is_crate => {
                try!(write!(fmt, "<a href='stability.html'>[stability]</a> "));
            }
            _ => {}
        };

        try!(write!(fmt,
        r##"<span id='render-detail'>
            <a id="collapse-all" href="#">[-]</a>&nbsp;<a id="expand-all" href="#">[+]</a>
        </span>"##));

        // Write `src` tag
        //
        // When this item is part of a `pub use` in a downstream crate, the
        // [src] link in the downstream documentation will actually come back to
        // this page, and this link will be auto-clicked. The `id` attribute is
        // used to find the link to auto-click.
        if self.cx.include_sources && !is_primitive {
            match self.href(self.cx) {
                Some(l) => {
                    try!(write!(fmt, "<a id='src-{}' href='{}'>[src]</a>",
                                self.item.def_id.node, l));
                }
                None => {}
            }
        }

        try!(write!(fmt, "</span>")); // out-of-band

        try!(write!(fmt, "</h1>\n"));

        match self.item.inner {
            clean::ModuleItem(ref m) => {
                item_module(fmt, self.cx, self.item, &m.items)
            }
            clean::FunctionItem(ref f) | clean::ForeignFunctionItem(ref f) =>
                item_function(fmt, self.item, f),
            clean::TraitItem(ref t) => item_trait(fmt, self.cx, self.item, t),
            clean::StructItem(ref s) => item_struct(fmt, self.item, s),
            clean::EnumItem(ref e) => item_enum(fmt, self.item, e),
            clean::TypedefItem(ref t) => item_typedef(fmt, self.item, t),
            clean::MacroItem(ref m) => item_macro(fmt, self.item, m),
            clean::PrimitiveItem(ref p) => item_primitive(fmt, self.item, p),
            clean::StaticItem(ref i) | clean::ForeignStaticItem(ref i) =>
                item_static(fmt, self.item, i),
            clean::ConstantItem(ref c) => item_constant(fmt, self.item, c),
            _ => Ok(())
        }
    }
}

fn item_path(item: &clean::Item) -> String {
    match item.inner {
        clean::ModuleItem(..) => {
            format!("{}/index.html", item.name.as_ref().unwrap())
        }
        _ => {
            format!("{}.{}.html",
                    shortty(item).to_static_str(),
                    *item.name.as_ref().unwrap())
        }
    }
}

fn full_path(cx: &Context, item: &clean::Item) -> String {
    let mut s = cx.current.connect("::");
    s.push_str("::");
    s.push_str(item.name.as_ref().unwrap());
    return s
}

fn shorter<'a>(s: Option<&'a str>) -> &'a str {
    match s {
        Some(s) => match s.find_str("\n\n") {
            Some(pos) => &s[..pos],
            None => s,
        },
        None => ""
    }
}

#[inline]
fn shorter_line(s: Option<&str>) -> String {
    shorter(s).replace("\n", " ")
}

fn document(w: &mut fmt::Formatter, item: &clean::Item) -> fmt::Result {
    match item.doc_value() {
        Some(s) => {
            try!(write!(w, "<div class='docblock'>{}</div>", Markdown(s)));
        }
        None => {}
    }
    Ok(())
}

fn item_module(w: &mut fmt::Formatter, cx: &Context,
               item: &clean::Item, items: &[clean::Item]) -> fmt::Result {
    try!(document(w, item));

    let mut indices = (0..items.len()).filter(|i| {
        !cx.ignore_private_item(&items[*i])
    }).collect::<Vec<uint>>();

    // the order of item types in the listing
    fn reorder(ty: ItemType) -> u8 {
        match ty {
            ItemType::ExternCrate     => 0,
            ItemType::Import          => 1,
            ItemType::Primitive       => 2,
            ItemType::Module          => 3,
            ItemType::Macro           => 4,
            ItemType::Struct          => 5,
            ItemType::Enum            => 6,
            ItemType::Constant        => 7,
            ItemType::Static          => 8,
            ItemType::Trait           => 9,
            ItemType::Function        => 10,
            ItemType::Typedef         => 12,
            _                         => 13 + ty as u8,
        }
    }

    fn cmp(i1: &clean::Item, i2: &clean::Item, idx1: uint, idx2: uint) -> Ordering {
        let ty1 = shortty(i1);
        let ty2 = shortty(i2);
        if ty1 == ty2 {
            return i1.name.cmp(&i2.name);
        }
        (reorder(ty1), idx1).cmp(&(reorder(ty2), idx2))
    }

    indices.sort_by(|&i1, &i2| cmp(&items[i1], &items[i2], i1, i2));

    debug!("{:?}", indices);
    let mut curty = None;
    for &idx in &indices {
        let myitem = &items[idx];

        let myty = Some(shortty(myitem));
        if curty == Some(ItemType::ExternCrate) && myty == Some(ItemType::Import) {
            // Put `extern crate` and `use` re-exports in the same section.
            curty = myty;
        } else if myty != curty {
            if curty.is_some() {
                try!(write!(w, "</table>"));
            }
            curty = myty;
            let (short, name) = match myty.unwrap() {
                ItemType::ExternCrate |
                ItemType::Import          => ("reexports", "Reexports"),
                ItemType::Module          => ("modules", "Modules"),
                ItemType::Struct          => ("structs", "Structs"),
                ItemType::Enum            => ("enums", "Enums"),
                ItemType::Function        => ("functions", "Functions"),
                ItemType::Typedef         => ("types", "Type Definitions"),
                ItemType::Static          => ("statics", "Statics"),
                ItemType::Constant        => ("constants", "Constants"),
                ItemType::Trait           => ("traits", "Traits"),
                ItemType::Impl            => ("impls", "Implementations"),
                ItemType::TyMethod        => ("tymethods", "Type Methods"),
                ItemType::Method          => ("methods", "Methods"),
                ItemType::StructField     => ("fields", "Struct Fields"),
                ItemType::Variant         => ("variants", "Variants"),
                ItemType::Macro           => ("macros", "Macros"),
                ItemType::Primitive       => ("primitives", "Primitive Types"),
                ItemType::AssociatedType  => ("associated-types", "Associated Types"),
            };
            try!(write!(w,
                        "<h2 id='{id}' class='section-header'>\
                        <a href=\"#{id}\">{name}</a></h2>\n<table>",
                        id = short, name = name));
        }

        match myitem.inner {
            clean::ExternCrateItem(ref name, ref src) => {
                match *src {
                    Some(ref src) => {
                        try!(write!(w, "<tr><td><code>{}extern crate \"{}\" as {};",
                                    VisSpace(myitem.visibility),
                                    src,
                                    name))
                    }
                    None => {
                        try!(write!(w, "<tr><td><code>{}extern crate {};",
                                    VisSpace(myitem.visibility), name))
                    }
                }
                try!(write!(w, "</code></td></tr>"));
            }

            clean::ImportItem(ref import) => {
                try!(write!(w, "<tr><td><code>{}{}</code></td></tr>",
                            VisSpace(myitem.visibility), *import));
            }

            _ => {
                if myitem.name.is_none() { continue }
                try!(write!(w, "
                    <tr>
                        <td>{stab}<a class='{class}' href='{href}'
                               title='{title}'>{}</a></td>
                        <td class='docblock short'>{}</td>
                    </tr>
                ",
                *myitem.name.as_ref().unwrap(),
                Markdown(shorter(myitem.doc_value())),
                class = shortty(myitem),
                href = item_path(myitem),
                title = full_path(cx, myitem),
                stab = ConciseStability(&myitem.stability)));
            }
        }
    }

    write!(w, "</table>")
}

struct Initializer<'a>(&'a str);

impl<'a> fmt::Display for Initializer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Initializer(s) = *self;
        if s.len() == 0 { return Ok(()); }
        try!(write!(f, "<code> = </code>"));
        write!(f, "<code>{}</code>", s)
    }
}

fn item_constant(w: &mut fmt::Formatter, it: &clean::Item,
                 c: &clean::Constant) -> fmt::Result {
    try!(write!(w, "<pre class='rust const'>{vis}const \
                    {name}: {typ}{init}</pre>",
           vis = VisSpace(it.visibility),
           name = it.name.as_ref().unwrap(),
           typ = c.type_,
           init = Initializer(&c.expr)));
    document(w, it)
}

fn item_static(w: &mut fmt::Formatter, it: &clean::Item,
               s: &clean::Static) -> fmt::Result {
    try!(write!(w, "<pre class='rust static'>{vis}static {mutability}\
                    {name}: {typ}{init}</pre>",
           vis = VisSpace(it.visibility),
           mutability = MutableSpace(s.mutability),
           name = it.name.as_ref().unwrap(),
           typ = s.type_,
           init = Initializer(&s.expr)));
    document(w, it)
}

fn item_function(w: &mut fmt::Formatter, it: &clean::Item,
                 f: &clean::Function) -> fmt::Result {
    try!(write!(w, "<pre class='rust fn'>{vis}{unsafety}fn \
                    {name}{generics}{decl}{where_clause}</pre>",
           vis = VisSpace(it.visibility),
           unsafety = UnsafetySpace(f.unsafety),
           name = it.name.as_ref().unwrap(),
           generics = f.generics,
           where_clause = WhereClause(&f.generics),
           decl = f.decl));
    document(w, it)
}

fn item_trait(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
              t: &clean::Trait) -> fmt::Result {
    let mut bounds = String::new();
    if t.bounds.len() > 0 {
        if bounds.len() > 0 {
            bounds.push(' ');
        }
        bounds.push_str(": ");
        for (i, p) in t.bounds.iter().enumerate() {
            if i > 0 { bounds.push_str(" + "); }
            bounds.push_str(&format!("{}", *p));
        }
    }

    // Output the trait definition
    try!(write!(w, "<pre class='rust trait'>{}{}trait {}{}{}{} ",
                  VisSpace(it.visibility),
                  UnsafetySpace(t.unsafety),
                  it.name.as_ref().unwrap(),
                  t.generics,
                  bounds,
                  WhereClause(&t.generics)));

    let types = t.items.iter().filter(|m| m.is_type()).collect::<Vec<_>>();
    let required = t.items.iter().filter(|m| m.is_req()).collect::<Vec<_>>();
    let provided = t.items.iter().filter(|m| m.is_def()).collect::<Vec<_>>();

    if t.items.len() == 0 {
        try!(write!(w, "{{ }}"));
    } else {
        try!(write!(w, "{{\n"));
        for t in &types {
            try!(write!(w, "    "));
            try!(render_method(w, t.item()));
            try!(write!(w, ";\n"));
        }
        if types.len() > 0 && required.len() > 0 {
            try!(w.write_str("\n"));
        }
        for m in &required {
            try!(write!(w, "    "));
            try!(render_method(w, m.item()));
            try!(write!(w, ";\n"));
        }
        if required.len() > 0 && provided.len() > 0 {
            try!(w.write_str("\n"));
        }
        for m in &provided {
            try!(write!(w, "    "));
            try!(render_method(w, m.item()));
            try!(write!(w, " {{ ... }}\n"));
        }
        try!(write!(w, "}}"));
    }
    try!(write!(w, "</pre>"));

    // Trait documentation
    try!(document(w, it));

    fn trait_item(w: &mut fmt::Formatter, m: &clean::TraitMethod)
                  -> fmt::Result {
        try!(write!(w, "<h3 id='{}.{}' class='method'>{}<code>",
                    shortty(m.item()),
                    *m.item().name.as_ref().unwrap(),
                    ConciseStability(&m.item().stability)));
        try!(render_method(w, m.item()));
        try!(write!(w, "</code></h3>"));
        try!(document(w, m.item()));
        Ok(())
    }

    if types.len() > 0 {
        try!(write!(w, "
            <h2 id='associated-types'>Associated Types</h2>
            <div class='methods'>
        "));
        for t in &types {
            try!(trait_item(w, *t));
        }
        try!(write!(w, "</div>"));
    }

    // Output the documentation for each function individually
    if required.len() > 0 {
        try!(write!(w, "
            <h2 id='required-methods'>Required Methods</h2>
            <div class='methods'>
        "));
        for m in &required {
            try!(trait_item(w, *m));
        }
        try!(write!(w, "</div>"));
    }
    if provided.len() > 0 {
        try!(write!(w, "
            <h2 id='provided-methods'>Provided Methods</h2>
            <div class='methods'>
        "));
        for m in &provided {
            try!(trait_item(w, *m));
        }
        try!(write!(w, "</div>"));
    }

    let cache = cache();
    try!(write!(w, "
        <h2 id='implementors'>Implementors</h2>
        <ul class='item-list' id='implementors-list'>
    "));
    match cache.implementors.get(&it.def_id) {
        Some(implementors) => {
            for i in implementors {
                try!(writeln!(w, "<li>{}<code>impl{} {} for {}{}</code></li>",
                              ConciseStability(&i.stability),
                              i.generics, i.trait_, i.for_, WhereClause(&i.generics)));
            }
        }
        None => {}
    }
    try!(write!(w, "</ul>"));
    try!(write!(w, r#"<script type="text/javascript" async
                              src="{root_path}/implementors/{path}/{ty}.{name}.js">
                      </script>"#,
                root_path = repeat("..").take(cx.current.len()).collect::<Vec<_>>().connect("/"),
                path = if ast_util::is_local(it.def_id) {
                    cx.current.connect("/")
                } else {
                    let path = &cache.external_paths[it.def_id];
                    path[..path.len() - 1].connect("/")
                },
                ty = shortty(it).to_static_str(),
                name = *it.name.as_ref().unwrap()));
    Ok(())
}

fn assoc_type(w: &mut fmt::Formatter, it: &clean::Item,
              typ: &clean::TyParam) -> fmt::Result {
    try!(write!(w, "type {}", it.name.as_ref().unwrap()));
    if typ.bounds.len() > 0 {
        try!(write!(w, ": {}", TyParamBounds(&*typ.bounds)))
    }
    if let Some(ref default) = typ.default {
        try!(write!(w, " = {}", default));
    }
    Ok(())
}

fn render_method(w: &mut fmt::Formatter, meth: &clean::Item) -> fmt::Result {
    fn method(w: &mut fmt::Formatter, it: &clean::Item,
              unsafety: ast::Unsafety, abi: abi::Abi,
              g: &clean::Generics, selfty: &clean::SelfTy,
              d: &clean::FnDecl) -> fmt::Result {
        use syntax::abi::Abi;

        write!(w, "{}{}fn <a href='#{ty}.{name}' class='fnname'>{name}</a>\
                   {generics}{decl}{where_clause}",
               match unsafety {
                   ast::Unsafety::Unsafe => "unsafe ",
                   _ => "",
               },
               match abi {
                   Abi::Rust => String::new(),
                   a => format!("extern {} ", a.to_string())
               },
               ty = shortty(it),
               name = it.name.as_ref().unwrap(),
               generics = *g,
               decl = Method(selfty, d),
               where_clause = WhereClause(g))
    }
    match meth.inner {
        clean::TyMethodItem(ref m) => {
            method(w, meth, m.unsafety, m.abi, &m.generics, &m.self_, &m.decl)
        }
        clean::MethodItem(ref m) => {
            method(w, meth, m.unsafety, m.abi, &m.generics, &m.self_, &m.decl)
        }
        clean::AssociatedTypeItem(ref typ) => {
            assoc_type(w, meth, typ)
        }
        _ => panic!("render_method called on non-method")
    }
}

fn item_struct(w: &mut fmt::Formatter, it: &clean::Item,
               s: &clean::Struct) -> fmt::Result {
    try!(write!(w, "<pre class='rust struct'>"));
    try!(render_attributes(w, it));
    try!(render_struct(w,
                       it,
                       Some(&s.generics),
                       s.struct_type,
                       &s.fields,
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
    if let doctree::Plain = s.struct_type {
        if fields.peek().is_some() {
            try!(write!(w, "<h2 class='fields'>Fields</h2>\n<table>"));
            for field in fields {
                try!(write!(w, "<tr><td id='structfield.{name}'>\
                                  {stab}<code>{name}</code></td><td>",
                            stab = ConciseStability(&field.stability),
                            name = field.name.as_ref().unwrap()));
                try!(document(w, field));
                try!(write!(w, "</td></tr>"));
            }
            try!(write!(w, "</table>"));
        }
    }
    render_methods(w, it)
}

fn item_enum(w: &mut fmt::Formatter, it: &clean::Item,
             e: &clean::Enum) -> fmt::Result {
    try!(write!(w, "<pre class='rust enum'>"));
    try!(render_attributes(w, it));
    try!(write!(w, "{}enum {}{}{}",
                  VisSpace(it.visibility),
                  it.name.as_ref().unwrap(),
                  e.generics,
                  WhereClause(&e.generics)));
    if e.variants.len() == 0 && !e.variants_stripped {
        try!(write!(w, " {{}}"));
    } else {
        try!(write!(w, " {{\n"));
        for v in &e.variants {
            try!(write!(w, "    "));
            let name = v.name.as_ref().unwrap();
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
                                               &s.fields,
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
        try!(write!(w, "}}"));
    }
    try!(write!(w, "</pre>"));

    try!(document(w, it));
    if e.variants.len() > 0 {
        try!(write!(w, "<h2 class='variants'>Variants</h2>\n<table>"));
        for variant in &e.variants {
            try!(write!(w, "<tr><td id='variant.{name}'>{stab}<code>{name}</code></td><td>",
                          stab = ConciseStability(&variant.stability),
                          name = variant.name.as_ref().unwrap()));
            try!(document(w, variant));
            match variant.inner {
                clean::VariantItem(ref var) => {
                    match var.kind {
                        clean::StructVariant(ref s) => {
                            let fields = s.fields.iter().filter(|f| {
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
                                              v = variant.name.as_ref().unwrap(),
                                              f = field.name.as_ref().unwrap()));
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

fn render_attributes(w: &mut fmt::Formatter, it: &clean::Item) -> fmt::Result {
    for attr in &it.attrs {
        match *attr {
            clean::Word(ref s) if *s == "must_use" => {
                try!(write!(w, "#[{}]\n", s));
            }
            clean::NameValue(ref k, ref v) if *k == "must_use" => {
                try!(write!(w, "#[{} = \"{}\"]\n", k, v));
            }
            _ => ()
        }
    }
    Ok(())
}

fn render_struct(w: &mut fmt::Formatter, it: &clean::Item,
                 g: Option<&clean::Generics>,
                 ty: doctree::StructType,
                 fields: &[clean::Item],
                 tab: &str,
                 structhead: bool) -> fmt::Result {
    try!(write!(w, "{}{}{}",
                  VisSpace(it.visibility),
                  if structhead {"struct "} else {""},
                  it.name.as_ref().unwrap()));
    match g {
        Some(g) => try!(write!(w, "{}{}", *g, WhereClause(g))),
        None => {}
    }
    match ty {
        doctree::Plain => {
            try!(write!(w, " {{\n{}", tab));
            let mut fields_stripped = false;
            for field in fields {
                match field.inner {
                    clean::StructFieldItem(clean::HiddenStructField) => {
                        fields_stripped = true;
                    }
                    clean::StructFieldItem(clean::TypedStructField(ref ty)) => {
                        try!(write!(w, "    {}{}: {},\n{}",
                                      VisSpace(field.visibility),
                                      field.name.as_ref().unwrap(),
                                      *ty,
                                      tab));
                    }
                    _ => unreachable!(),
                };
            }

            if fields_stripped {
                try!(write!(w, "    // some fields omitted\n{}", tab));
            }
            try!(write!(w, "}}"));
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

fn render_methods(w: &mut fmt::Formatter, it: &clean::Item) -> fmt::Result {
    match cache().impls.get(&it.def_id) {
        Some(v) => {
            let (non_trait, traits): (Vec<_>, _) = v.iter().cloned()
                .partition(|i| i.impl_.trait_.is_none());
            if non_trait.len() > 0 {
                try!(write!(w, "<h2 id='methods'>Methods</h2>"));
                for i in &non_trait {
                    try!(render_impl(w, i));
                }
            }
            if traits.len() > 0 {
                try!(write!(w, "<h2 id='implementations'>Trait \
                                  Implementations</h2>"));
                let (derived, manual): (Vec<_>, _) = traits.into_iter()
                    .partition(|i| i.impl_.derived);
                for i in &manual {
                    try!(render_impl(w, i));
                }
                if derived.len() > 0 {
                    try!(write!(w, "<h3 id='derived_implementations'>Derived Implementations \
                                </h3>"));
                    for i in &derived {
                        try!(render_impl(w, i));
                    }
                }
            }
        }
        None => {}
    }
    Ok(())
}

fn render_impl(w: &mut fmt::Formatter, i: &Impl) -> fmt::Result {
    try!(write!(w, "<h3 class='impl'>{}<code>impl{} ",
                ConciseStability(&i.stability),
                i.impl_.generics));
    match i.impl_.polarity {
        Some(clean::ImplPolarity::Negative) => try!(write!(w, "!")),
        _ => {}
    }
    match i.impl_.trait_ {
        Some(ref ty) => try!(write!(w, "{} for ", *ty)),
        None => {}
    }
    try!(write!(w, "{}{}</code></h3>", i.impl_.for_, WhereClause(&i.impl_.generics)));
    match i.dox {
        Some(ref dox) => {
            try!(write!(w, "<div class='docblock'>{}</div>",
                          Markdown(dox)));
        }
        None => {}
    }

    fn doctraititem(w: &mut fmt::Formatter, item: &clean::Item, dox: bool)
                    -> fmt::Result {
        match item.inner {
            clean::MethodItem(..) | clean::TyMethodItem(..) => {
                try!(write!(w, "<h4 id='method.{}' class='{}'>{}<code>",
                            *item.name.as_ref().unwrap(),
                            shortty(item),
                            ConciseStability(&item.stability)));
                try!(render_method(w, item));
                try!(write!(w, "</code></h4>\n"));
            }
            clean::TypedefItem(ref tydef) => {
                let name = item.name.as_ref().unwrap();
                try!(write!(w, "<h4 id='assoc_type.{}' class='{}'>{}<code>",
                            *name,
                            shortty(item),
                            ConciseStability(&item.stability)));
                try!(write!(w, "type {} = {}", name, tydef.type_));
                try!(write!(w, "</code></h4>\n"));
            }
            clean::AssociatedTypeItem(ref typaram) => {
                let name = item.name.as_ref().unwrap();
                try!(write!(w, "<h4 id='assoc_type.{}' class='{}'>{}<code>",
                            *name,
                            shortty(item),
                            ConciseStability(&item.stability)));
                try!(assoc_type(w, item, typaram));
                try!(write!(w, "</code></h4>\n"));
            }
            _ => panic!("can't make docs for trait item with name {:?}", item.name)
        }
        match item.doc_value() {
            Some(s) if dox => {
                try!(write!(w, "<div class='docblock'>{}</div>", Markdown(s)));
                Ok(())
            }
            Some(..) | None => Ok(())
        }
    }

    try!(write!(w, "<div class='impl-items'>"));
    for trait_item in &i.impl_.items {
        try!(doctraititem(w, trait_item, true));
    }

    fn render_default_methods(w: &mut fmt::Formatter,
                              t: &clean::Trait,
                              i: &clean::Impl) -> fmt::Result {
        for trait_item in &t.items {
            let n = trait_item.item().name.clone();
            match i.items.iter().find(|m| { m.name == n }) {
                Some(..) => continue,
                None => {}
            }

            try!(doctraititem(w, trait_item.item(), false));
        }
        Ok(())
    }

    // If we've implemented a trait, then also emit documentation for all
    // default methods which weren't overridden in the implementation block.
    // FIXME: this also needs to be done for associated types, whenever defaults
    // for them work.
    match i.impl_.trait_ {
        Some(clean::ResolvedPath { did, .. }) => {
            try!({
                match cache().traits.get(&did) {
                    Some(t) => try!(render_default_methods(w, t, &i.impl_)),
                    None => {}
                }
                Ok(())
            })
        }
        Some(..) | None => {}
    }
    try!(write!(w, "</div>"));
    Ok(())
}

fn item_typedef(w: &mut fmt::Formatter, it: &clean::Item,
                t: &clean::Typedef) -> fmt::Result {
    try!(write!(w, "<pre class='rust typedef'>type {}{} = {};</pre>",
                  it.name.as_ref().unwrap(),
                  t.generics,
                  t.type_));

    document(w, it)
}

impl<'a> fmt::Display for Sidebar<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let cx = self.cx;
        let it = self.item;
        try!(write!(fmt, "<p class='location'>"));
        let len = cx.current.len() - if it.is_mod() {1} else {0};
        for (i, name) in cx.current.iter().take(len).enumerate() {
            if i > 0 {
                try!(write!(fmt, "::<wbr>"));
            }
            try!(write!(fmt, "<a href='{}index.html'>{}</a>",
                          &cx.root_path[..(cx.current.len() - i - 1) * 3],
                          *name));
        }
        try!(write!(fmt, "</p>"));

        fn block(w: &mut fmt::Formatter, short: &str, longty: &str,
                 cur: &clean::Item, cx: &Context) -> fmt::Result {
            let items = match cx.sidebar.get(short) {
                Some(items) => items,
                None => return Ok(())
            };
            try!(write!(w, "<div class='block {}'><h2>{}</h2>", short, longty));
            for &NameDoc(ref name, ref doc) in items {
                let curty = shortty(cur).to_static_str();
                let class = if cur.name.as_ref().unwrap() == name &&
                               short == curty { "current" } else { "" };
                try!(write!(w, "<a class='{ty} {class}' href='{href}{path}' \
                                title='{title}'>{name}</a>",
                       ty = short,
                       class = class,
                       href = if curty == "mod" {"../"} else {""},
                       path = if short == "mod" {
                           format!("{}/index.html", name)
                       } else {
                           format!("{}.{}.html", short, name)
                       },
                       title = Escape(doc.as_ref().unwrap()),
                       name = name));
            }
            try!(write!(w, "</div>"));
            Ok(())
        }

        try!(block(fmt, "mod", "Modules", it, cx));
        try!(block(fmt, "struct", "Structs", it, cx));
        try!(block(fmt, "enum", "Enums", it, cx));
        try!(block(fmt, "trait", "Traits", it, cx));
        try!(block(fmt, "fn", "Functions", it, cx));
        try!(block(fmt, "macro", "Macros", it, cx));
        Ok(())
    }
}

impl<'a> fmt::Display for Source<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Source(s) = *self;
        let lines = s.lines().count();
        let mut cols = 0;
        let mut tmp = lines;
        while tmp > 0 {
            cols += 1;
            tmp /= 10;
        }
        try!(write!(fmt, "<pre class=\"line-numbers\">"));
        for i in 1..lines + 1 {
            try!(write!(fmt, "<span id=\"{0}\">{0:1$}</span>\n", i, cols));
        }
        try!(write!(fmt, "</pre>"));
        try!(write!(fmt, "{}", highlight::highlight(s, None, None)));
        Ok(())
    }
}

fn item_macro(w: &mut fmt::Formatter, it: &clean::Item,
              t: &clean::Macro) -> fmt::Result {
    try!(w.write_str(&highlight::highlight(&t.source,
                                          Some("macro"),
                                          None)));
    document(w, it)
}

fn item_primitive(w: &mut fmt::Formatter,
                  it: &clean::Item,
                  _p: &clean::PrimitiveType) -> fmt::Result {
    try!(document(w, it));
    render_methods(w, it)
}

fn get_basic_keywords() -> &'static str {
    "rust, rustlang, rust-lang"
}

fn make_item_keywords(it: &clean::Item) -> String {
    format!("{}, {}", get_basic_keywords(), it.name.as_ref().unwrap())
}

pub fn cache() -> Arc<Cache> {
    CACHE_KEY.with(|c| c.borrow().clone())
}
