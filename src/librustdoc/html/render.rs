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

use std::collections::{HashMap, HashSet};
use std::collections::hashmap::{Occupied, Vacant};
use std::fmt;
use std::io::fs::PathExtensions;
use std::io::{fs, File, BufferedWriter, MemWriter, BufferedReader};
use std::io;
use std::str;
use std::string::String;
use std::sync::Arc;

use externalfiles::ExternalHtml;

use serialize::json;
use serialize::Encodable;
use serialize::json::ToJson;
use syntax::ast;
use syntax::ast_util;
use rustc::util::nodemap::NodeSet;

use clean;
use doctree;
use fold::DocFolder;
use html::format::{VisSpace, Method, FnStyleSpace, MutableSpace, Stability};
use html::format::{ConciseStability, WhereClause};
use html::highlight;
use html::item_type::{ItemType, shortty};
use html::item_type;
use html::layout;
use html::markdown::Markdown;
use html::markdown;
use stability_summary;

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
    pub current: Vec<String>,
    /// String representation of how to get back to the root path of the 'doc/'
    /// folder in terms of a relative URL.
    pub root_path: String,
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
    pub sidebar: HashMap<String, Vec<String>>,
    /// This flag indicates whether [src] links should be generated or not. If
    /// the source files are present in the html rendering, then this will be
    /// `true`.
    pub include_sources: bool,
    /// A flag, which when turned off, will render pages which redirect to the
    /// real location of an item. This is used to allow external links to
    /// publicly reused items to redirect to the right location.
    pub render_redirect_pages: bool,
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
    def_id: ast::DefId,
    generics: clean::Generics,
    trait_: clean::Type,
    for_: clean::Type,
    stability: Option<clean::Stability>,
}

/// Metadata about implementations for a type.
#[deriving(Clone)]
pub struct Impl {
    impl_: clean::Impl,
    dox: Option<String>,
    stability: Option<clean::Stability>,
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

struct Item<'a> { cx: &'a Context, item: &'a clean::Item, }
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

local_data_key!(pub cache_key: Arc<Cache>)
local_data_key!(pub current_location_key: Vec<String> )

/// Generates the documentation for `crate` into the directory `dst`
pub fn run(mut krate: clean::Crate, external_html: &ExternalHtml, dst: Path) -> io::IoResult<()> {
    let mut cx = Context {
        dst: dst,
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

    // Crawl the crate, building a summary of the stability levels.  NOTE: this
    // summary *must* be computed with the original `krate`; the folding below
    // removes the impls from their modules.
    let summary = stability_summary::build(&krate);

    // Crawl the crate attributes looking for attributes which control how we're
    // going to emit HTML
    let default: &[_] = &[];
    match krate.module.as_ref().map(|m| m.doc_list().unwrap_or(default)) {
        Some(attrs) => {
            for attr in attrs.iter() {
                match *attr {
                    clean::NameValue(ref x, ref s)
                            if "html_favicon_url" == x.as_slice() => {
                        cx.layout.favicon = s.to_string();
                    }
                    clean::NameValue(ref x, ref s)
                            if "html_logo_url" == x.as_slice() => {
                        cx.layout.logo = s.to_string();
                    }
                    clean::NameValue(ref x, ref s)
                            if "html_playground_url" == x.as_slice() => {
                        cx.layout.playground_url = s.to_string();
                        let name = krate.name.clone();
                        if markdown::playground_krate.get().is_none() {
                            markdown::playground_krate.replace(Some(Some(name)));
                        }
                    }
                    clean::Word(ref x)
                            if "html_no_source" == x.as_slice() => {
                        cx.include_sources = false;
                    }
                    _ => {}
                }
            }
        }
        None => {}
    }

    // Crawl the crate to build various caches used for the output
    let analysis = ::analysiskey.get();
    let public_items = analysis.as_ref().map(|a| a.public_items.clone());
    let public_items = public_items.unwrap_or(NodeSet::new());
    let paths: HashMap<ast::DefId, (Vec<String>, ItemType)> =
      analysis.as_ref().map(|a| {
        let paths = a.external_paths.borrow_mut().take().unwrap();
        paths.into_iter().map(|(k, (v, t))| {
            (k, (v, match t {
                clean::TypeStruct => item_type::Struct,
                clean::TypeEnum => item_type::Enum,
                clean::TypeFunction => item_type::Function,
                clean::TypeTrait => item_type::Trait,
                clean::TypeModule => item_type::Module,
                clean::TypeStatic => item_type::Static,
                clean::TypeVariant => item_type::Variant,
                clean::TypeTypedef => item_type::Typedef,
            }))
        }).collect()
    }).unwrap_or(HashMap::new());
    let mut cache = Cache {
        impls: HashMap::new(),
        external_paths: paths.iter().map(|(&k, v)| (k, v.ref0().clone()))
                             .collect(),
        paths: paths,
        implementors: HashMap::new(),
        stack: Vec::new(),
        parent_stack: Vec::new(),
        search_index: Vec::new(),
        extern_locations: HashMap::new(),
        primitive_locations: HashMap::new(),
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
    for &(n, ref e) in krate.externs.iter() {
        cache.extern_locations.insert(n, extern_location(e, &cx.dst));
        let did = ast::DefId { krate: n, node: ast::CRATE_NODE_ID };
        cache.paths.insert(did, (vec![e.name.to_string()], item_type::Module));
    }

    // Cache where all known primitives have their documentation located.
    //
    // Favor linking to as local extern as possible, so iterate all crates in
    // reverse topological order.
    for &(n, ref e) in krate.externs.iter().rev() {
        for &prim in e.primitives.iter() {
            cache.primitive_locations.insert(prim, n);
        }
    }
    for &prim in krate.primitives.iter() {
        cache.primitive_locations.insert(prim, ast::LOCAL_CRATE);
    }

    // Build our search index
    let index = try!(build_index(&krate, &mut cache));

    // Freeze the cache now that the index has been built. Put an Arc into TLS
    // for future parallelization opportunities
    let cache = Arc::new(cache);
    cache_key.replace(Some(cache.clone()));
    current_location_key.replace(Some(Vec::new()));

    try!(write_shared(&cx, &krate, &*cache, index));
    let krate = try!(render_sources(&mut cx, krate));

    // And finally render the whole crate's documentation
    cx.krate(krate, summary)
}

fn build_index(krate: &clean::Crate, cache: &mut Cache) -> io::IoResult<String> {
    // Build the search index from the collected metadata
    let mut nodeid_to_pathid = HashMap::new();
    let mut pathid_to_nodeid = Vec::new();
    {
        let Cache { ref mut search_index,
                    ref orphan_methods,
                    ref mut paths, .. } = *cache;

        // Attach all orphan methods to the type's definition if the type
        // has since been learned.
        for &(pid, ref item) in orphan_methods.iter() {
            let did = ast_util::local_def(pid);
            match paths.find(&did) {
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
        for item in search_index.iter() {
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
    let mut w = MemWriter::new();
    try!(write!(&mut w, r#"searchIndex['{}'] = {{"items":["#, krate.name));

    let mut lastpath = "".to_string();
    for (i, item) in cache.search_index.iter().enumerate() {
        // Omit the path if it is same to that of the prior item.
        let path;
        if lastpath.as_slice() == item.path.as_slice() {
            path = "";
        } else {
            lastpath = item.path.to_string();
            path = item.path.as_slice();
        };

        if i > 0 {
            try!(write!(&mut w, ","));
        }
        try!(write!(&mut w, r#"[{:u},"{}","{}",{}"#,
                    item.ty, item.name, path,
                    item.desc.to_json().to_string()));
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

    for (i, &did) in pathid_to_nodeid.iter().enumerate() {
        let &(ref fqp, short) = cache.paths.find(&did).unwrap();
        if i > 0 {
            try!(write!(&mut w, ","));
        }
        try!(write!(&mut w, r#"[{:u},"{}"]"#,
                    short, *fqp.last().unwrap()));
    }

    try!(write!(&mut w, "]}};"));

    Ok(String::from_utf8(w.unwrap()).unwrap())
}

fn write_shared(cx: &Context,
                krate: &clean::Crate,
                cache: &Cache,
                search_index: String) -> io::IoResult<()> {
    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    try!(mkdir(&cx.dst));
    let _lock = ::flock::Lock::new(&cx.dst.join(".lock"));

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.
    try!(write(cx.dst.join("jquery.js"),
               include_bin!("static/jquery-2.1.0.min.js")));
    try!(write(cx.dst.join("main.js"), include_bin!("static/main.js")));
    try!(write(cx.dst.join("playpen.js"), include_bin!("static/playpen.js")));
    try!(write(cx.dst.join("main.css"), include_bin!("static/main.css")));
    try!(write(cx.dst.join("normalize.css"),
               include_bin!("static/normalize.css")));
    try!(write(cx.dst.join("FiraSans-Regular.woff"),
               include_bin!("static/FiraSans-Regular.woff")));
    try!(write(cx.dst.join("FiraSans-Medium.woff"),
               include_bin!("static/FiraSans-Medium.woff")));
    try!(write(cx.dst.join("Heuristica-Italic.woff"),
               include_bin!("static/Heuristica-Italic.woff")));
    try!(write(cx.dst.join("SourceSerifPro-Regular.woff"),
               include_bin!("static/SourceSerifPro-Regular.woff")));
    try!(write(cx.dst.join("SourceSerifPro-Bold.woff"),
               include_bin!("static/SourceSerifPro-Bold.woff")));
    try!(write(cx.dst.join("SourceCodePro-Regular.woff"),
               include_bin!("static/SourceCodePro-Regular.woff")));
    try!(write(cx.dst.join("SourceCodePro-Semibold.woff"),
               include_bin!("static/SourceCodePro-Semibold.woff")));

    fn collect(path: &Path, krate: &str,
               key: &str) -> io::IoResult<Vec<String>> {
        let mut ret = Vec::new();
        if path.exists() {
            for line in BufferedReader::new(File::open(path)).lines() {
                let line = try!(line);
                if !line.as_slice().starts_with(key) {
                    continue
                }
                if line.as_slice().starts_with(
                        format!("{}['{}']", key, krate).as_slice()) {
                    continue
                }
                ret.push(line.to_string());
            }
        }
        return Ok(ret);
    }

    // Update the search index
    let dst = cx.dst.join("search-index.js");
    let all_indexes = try!(collect(&dst, krate.name.as_slice(),
                                   "searchIndex"));
    let mut w = try!(File::create(&dst));
    try!(writeln!(&mut w, "var searchIndex = {{}};"));
    try!(writeln!(&mut w, "{}", search_index));
    for index in all_indexes.iter() {
        try!(writeln!(&mut w, "{}", *index));
    }
    try!(writeln!(&mut w, "initSearch(searchIndex);"));

    // Update the list of all implementors for traits
    let dst = cx.dst.join("implementors");
    try!(mkdir(&dst));
    for (&did, imps) in cache.implementors.iter() {
        // Private modules can leak through to this phase of rustdoc, which
        // could contain implementations for otherwise private types. In some
        // rare cases we could find an implementation for an item which wasn't
        // indexed, so we just skip this step in that case.
        //
        // FIXME: this is a vague explanation for why this can't be a `get`, in
        //        theory it should be...
        let &(ref remote_path, remote_item_type) = match cache.paths.find(&did) {
            Some(p) => p,
            None => continue,
        };

        let mut mydst = dst.clone();
        for part in remote_path[..remote_path.len() - 1].iter() {
            mydst.push(part.as_slice());
            try!(mkdir(&mydst));
        }
        mydst.push(format!("{}.{}.js",
                           remote_item_type.to_static_str(),
                           remote_path[remote_path.len() - 1]));
        let all_implementors = try!(collect(&mydst, krate.name.as_slice(),
                                            "implementors"));

        try!(mkdir(&mydst.dir_path()));
        let mut f = BufferedWriter::new(try!(File::create(&mydst)));
        try!(writeln!(&mut f, "(function() {{var implementors = {{}};"));

        for implementor in all_implementors.iter() {
            try!(write!(&mut f, "{}", *implementor));
        }

        try!(write!(&mut f, r"implementors['{}'] = [", krate.name));
        for imp in imps.iter() {
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
                  krate: clean::Crate) -> io::IoResult<clean::Crate> {
    info!("emitting source files");
    let dst = cx.dst.join("src");
    try!(mkdir(&dst));
    let dst = dst.join(krate.name.as_slice());
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
fn write(dst: Path, contents: &[u8]) -> io::IoResult<()> {
    File::create(&dst).write(contents)
}

/// Makes a directory on the filesystem, failing the task if an error occurs and
/// skipping if the directory already exists.
fn mkdir(path: &Path) -> io::IoResult<()> {
    if !path.exists() {
        fs::mkdir(path, io::USER_RWX)
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
    if p.as_vec() != b"." {
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
            clean::List(ref x, ref list) if "doc" == x.as_slice() => {
                for attr in list.iter() {
                    match *attr {
                        clean::NameValue(ref x, ref s)
                                if "html_root_url" == x.as_slice() => {
                            if s.as_slice().ends_with("/") {
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
                .include_sources = match self.emit_source(item.source
                                                              .filename
                                                              .as_slice()) {
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
            contents.slice_from(3)
        } else {
            contents
        };

        // Create the intermediate directories
        let mut cur = self.dst.clone();
        let mut root_path = String::from_str("../../");
        clean_srcpath(p.dirname(), |component| {
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
            title: title.as_slice(),
            ty: "source",
            root_path: root_path.as_slice(),
            description: desc.as_slice(),
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
                self.traits.insert(item.def_id, t.clone());
            }
            _ => {}
        }

        // Collect all the implementors of traits.
        match item.inner {
            clean::ImplItem(ref i) => {
                match i.trait_ {
                    Some(clean::ResolvedPath{ did, .. }) => {
                        let v = match self.implementors.entry(did) {
                            Vacant(entry) => entry.set(Vec::with_capacity(1)),
                            Occupied(entry) => entry.into_mut(),
                        };
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
            _ => {}
        }

        // Index this method for searching later on
        match item.name {
            Some(ref s) => {
                let (parent, is_method) = match item.inner {
                    clean::TyMethodItem(..) |
                    clean::StructFieldItem(..) |
                    clean::VariantItem(..) => {
                        ((Some(*self.parent_stack.last().unwrap()),
                          Some(self.stack[..self.stack.len() - 1])),
                          false)
                    }
                    clean::MethodItem(..) => {
                        if self.parent_stack.len() == 0 {
                            ((None, None), false)
                        } else {
                            let last = self.parent_stack.last().unwrap();
                            let did = *last;
                            let path = match self.paths.find(&did) {
                                Some(&(_, item_type::Trait)) =>
                                    Some(self.stack[..self.stack.len() - 1]),
                                // The current stack not necessarily has correlation for
                                // where the type was defined. On the other hand,
                                // `paths` always has the right information if present.
                                Some(&(ref fqp, item_type::Struct)) |
                                Some(&(ref fqp, item_type::Enum)) =>
                                    Some(fqp[..fqp.len() - 1]),
                                Some(..) => Some(self.stack.as_slice()),
                                None => None
                            };
                            ((Some(*last), path), true)
                        }
                    }
                    _ => ((None, Some(self.stack.as_slice())), false)
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
            None => {}
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
                self.paths.insert(item.def_id, (stack, item_type::Enum));
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
                                        if "doc" == x.as_slice() => {
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

                        match did {
                            Some(did) => {
                                let v = match self.impls.entry(did) {
                                    Vacant(entry) => entry.set(Vec::with_capacity(1)),
                                    Occupied(entry) => entry.into_mut(),
                                };
                                v.push(Impl {
                                    impl_: i,
                                    dox: dox,
                                    stability: item.stability.clone(),
                                });
                            }
                            None => {}
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
        for typ in generics.type_params.iter() {
            self.typarams.insert(typ.did, typ.name.clone());
        }
    }
}

impl Context {
    /// Recurse in the directory structure and change the "root path" to make
    /// sure it always points to the top (relatively)
    fn recurse<T>(&mut self, s: String, f: |&mut Context| -> T) -> T {
        if s.len() == 0 {
            fail!("Unexpected empty destination: {}", self.current);
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
    fn krate(mut self, mut krate: clean::Crate,
             stability: stability_summary::ModuleSummary) -> io::IoResult<()> {
        let mut item = match krate.module.take() {
            Some(i) => i,
            None => return Ok(())
        };
        item.name = Some(krate.name);

        // render stability dashboard
        try!(self.recurse(stability.name.clone(), |this| {
            let json_dst = &this.dst.join("stability.json");
            let mut json_out = BufferedWriter::new(try!(File::create(json_dst)));
            try!(stability.encode(&mut json::Encoder::new(&mut json_out)));

            let mut title = stability.name.clone();
            title.push_str(" - Stability dashboard");
            let desc = format!("API stability overview for the Rust `{}` crate.",
                               this.layout.krate);
            let page = layout::Page {
                ty: "mod",
                root_path: this.root_path.as_slice(),
                title: title.as_slice(),
                description: desc.as_slice(),
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
    fn item(&mut self, item: clean::Item,
            f: |&mut Context, clean::Item|) -> io::IoResult<()> {
        fn render(w: io::File, cx: &Context, it: &clean::Item,
                  pushname: bool) -> io::IoResult<()> {
            info!("Rendering an item to {}", w.path().display());
            // A little unfortunate that this is done like this, but it sure
            // does make formatting *a lot* nicer.
            current_location_key.replace(Some(cx.current.clone()));

            let mut title = cx.current.connect("::");
            if pushname {
                if title.len() > 0 {
                    title.push_str("::");
                }
                title.push_str(it.name.as_ref().unwrap().as_slice());
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
                        it.name.get_ref(), tyname, cx.layout.krate)
            };
            let keywords = make_item_keywords(it);
            let page = layout::Page {
                ty: tyname,
                root_path: cx.root_path.as_slice(),
                title: title.as_slice(),
                description: desc.as_slice(),
                keywords: keywords.as_slice(),
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
                let mut url = "../".repeat(cx.current.len());
                match cache_key.get().unwrap().paths.find(&it.def_id) {
                    Some(&(ref names, _)) => {
                        for name in names[..names.len() - 1].iter() {
                            url.push_str(name.as_slice());
                            url.push_str("/");
                        }
                        url.push_str(item_path(it).as_slice());
                        try!(layout::redirect(&mut writer, url.as_slice()));
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
            self.render_redirect_pages = ignore_private_item(&item);
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
                    this.sidebar = build_sidebar(&m);
                    for item in m.items.into_iter() {
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

    /// Generate a url appropriate for an `href` attribute back to the source of
    /// this item.
    ///
    /// The url generated, when clicked, will redirect the browser back to the
    /// original source code.
    ///
    /// If `None` is returned, then a source link couldn't be generated. This
    /// may happen, for example, with externally inlined items where the source
    /// of their crate documentation isn't known.
    fn href(&self) -> Option<String> {
        // If this item is part of the local crate, then we're guaranteed to
        // know the span, so we plow forward and generate a proper url. The url
        // has anchors for the line numbers that we're linking to.
        if ast_util::is_local(self.item.def_id) {
            let mut path = Vec::new();
            clean_srcpath(self.item.source.filename.as_bytes(), |component| {
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
            let cache = cache_key.get().unwrap();
            let path = &cache.external_paths[self.item.def_id];
            let root = match cache.extern_locations[self.item.def_id.krate] {
                Remote(ref s) => s.to_string(),
                Local => self.cx.root_path.clone(),
                Unknown => return None,
            };
            Some(format!("{root}{path}/{file}?gotosrc={goto}",
                         root = root,
                         path = path.slice_to(path.len() - 1).connect("/"),
                         file = item_path(self.item),
                         goto = self.item.def_id.node))
        }
    }
}



impl<'a> fmt::Show for Item<'a> {
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
            let cur = self.cx.current.as_slice();
            let amt = if self.ismodule() { cur.len() - 1 } else { cur.len() };
            for (i, component) in cur.iter().enumerate().take(amt) {
                try!(write!(fmt, "<a href='{}index.html'>{}</a>::<wbr>",
                            "../".repeat(cur.len() - i - 1),
                            component.as_slice()));
            }
        }
        try!(write!(fmt, "<a class='{}' href=''>{}</a>",
                    shortty(self.item), self.item.name.get_ref().as_slice()));

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
            <a id="collapse-all" href="#">[-]
            </a>&nbsp;<a id="expand-all" href="#">[+]</a>
        </span>"##));

        // Write `src` tag
        //
        // When this item is part of a `pub use` in a downstream crate, the
        // [src] link in the downstream documentation will actually come back to
        // this page, and this link will be auto-clicked. The `id` attribute is
        // used to find the link to auto-click.
        if self.cx.include_sources && !is_primitive {
            match self.href() {
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
                item_module(fmt, self.cx, self.item, m.items.as_slice())
            }
            clean::FunctionItem(ref f) | clean::ForeignFunctionItem(ref f) =>
                item_function(fmt, self.item, f),
            clean::TraitItem(ref t) => item_trait(fmt, self.cx, self.item, t),
            clean::StructItem(ref s) => item_struct(fmt, self.item, s),
            clean::EnumItem(ref e) => item_enum(fmt, self.item, e),
            clean::TypedefItem(ref t) => item_typedef(fmt, self.item, t),
            clean::MacroItem(ref m) => item_macro(fmt, self.item, m),
            clean::PrimitiveItem(ref p) => item_primitive(fmt, self.item, p),
            _ => Ok(())
        }
    }
}

fn item_path(item: &clean::Item) -> String {
    match item.inner {
        clean::ModuleItem(..) => {
            format!("{}/index.html", item.name.get_ref())
        }
        _ => {
            format!("{}.{}.html",
                    shortty(item).to_static_str(),
                    *item.name.get_ref())
        }
    }
}

fn full_path(cx: &Context, item: &clean::Item) -> String {
    let mut s = cx.current.connect("::");
    s.push_str("::");
    s.push_str(item.name.as_ref().unwrap().as_slice());
    return s
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

    let mut indices = range(0, items.len()).filter(|i| {
        !ignore_private_item(&items[*i])
    }).collect::<Vec<uint>>();

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
            (&clean::PrimitiveItem(..), _) => Less,
            (_, &clean::PrimitiveItem(..)) => Greater,
            (&clean::ModuleItem(..), _) => Less,
            (_, &clean::ModuleItem(..)) => Greater,
            (&clean::MacroItem(..), _) => Less,
            (_, &clean::MacroItem(..)) => Greater,
            (&clean::StructItem(..), _) => Less,
            (_, &clean::StructItem(..)) => Greater,
            (&clean::EnumItem(..), _) => Less,
            (_, &clean::EnumItem(..)) => Greater,
            (&clean::ConstantItem(..), _) => Less,
            (_, &clean::ConstantItem(..)) => Greater,
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

    indices.sort_by(|&i1, &i2| cmp(&items[i1], &items[i2], i1, i2));

    debug!("{}", indices);
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
                clean::ConstantItem(..)        => ("constants", "Constants"),
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
                clean::PrimitiveItem(..)       => ("primitives", "Primitive Types"),
                clean::AssociatedTypeItem(..)  => ("associated-types", "Associated Types"),
            };
            try!(write!(w,
                        "<h2 id='{id}' class='section-header'>\
                        <a href=\"#{id}\">{name}</a></h2>\n<table>",
                        id = short, name = name));
        }

        struct Initializer<'a>(&'a str, Item<'a>);
        impl<'a> fmt::Show for Initializer<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let Initializer(s, item) = *self;
                if s.len() == 0 { return Ok(()); }
                try!(write!(f, "<code> = </code>"));
                if s.contains("\n") {
                    match item.href() {
                        Some(url) => {
                            write!(f, "<a href='{}'>[definition]</a>",
                                   url)
                        }
                        None => Ok(()),
                    }
                } else {
                    write!(f, "<code>{}</code>", s.as_slice())
                }
            }
        }

        match myitem.inner {
            clean::StaticItem(ref s) | clean::ForeignStaticItem(ref s) => {
                try!(write!(w, "
                    <tr>
                        <td>{}<code>{}static {}{}: {}</code>{}</td>
                        <td class='docblock'>{}&nbsp;</td>
                    </tr>
                ",
                ConciseStability(&myitem.stability),
                VisSpace(myitem.visibility),
                MutableSpace(s.mutability),
                *myitem.name.get_ref(),
                s.type_,
                Initializer(s.expr.as_slice(), Item { cx: cx, item: myitem }),
                Markdown(blank(myitem.doc_value()))));
            }
            clean::ConstantItem(ref s) => {
                try!(write!(w, "
                    <tr>
                        <td>{}<code>{}const {}: {}</code>{}</td>
                        <td class='docblock'>{}&nbsp;</td>
                    </tr>
                ",
                ConciseStability(&myitem.stability),
                VisSpace(myitem.visibility),
                *myitem.name.get_ref(),
                s.type_,
                Initializer(s.expr.as_slice(), Item { cx: cx, item: myitem }),
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
                        <td>{stab}<a class='{class}' href='{href}'
                               title='{title}'>{}</a></td>
                        <td class='docblock short'>{}</td>
                    </tr>
                ",
                *myitem.name.get_ref(),
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

fn item_function(w: &mut fmt::Formatter, it: &clean::Item,
                 f: &clean::Function) -> fmt::Result {
    try!(write!(w, "<pre class='rust fn'>{vis}{fn_style}fn \
                    {name}{generics}{decl}{where_clause}</pre>",
           vis = VisSpace(it.visibility),
           fn_style = FnStyleSpace(f.fn_style),
           name = it.name.get_ref().as_slice(),
           generics = f.generics,
           where_clause = WhereClause(&f.generics),
           decl = f.decl));
    document(w, it)
}

fn item_trait(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
              t: &clean::Trait) -> fmt::Result {
    let mut bounds = String::new();
    if t.bounds.len() > 0 {
        bounds.push_str(": ");
        for (i, p) in t.bounds.iter().enumerate() {
            if i > 0 { bounds.push_str(" + "); }
            bounds.push_str(format!("{}", *p).as_slice());
        }
    }

    // Output the trait definition
    try!(write!(w, "<pre class='rust trait'>{}trait {}{}{}{} ",
                  VisSpace(it.visibility),
                  it.name.get_ref().as_slice(),
                  t.generics,
                  bounds,
                  WhereClause(&t.generics)));
    let required = t.items.iter()
                          .filter(|m| {
                              match **m {
                                  clean::RequiredMethod(_) => true,
                                  _ => false,
                              }
                          })
                          .collect::<Vec<&clean::TraitMethod>>();
    let provided = t.items.iter()
                          .filter(|m| {
                              match **m {
                                  clean::ProvidedMethod(_) => true,
                                  _ => false,
                              }
                          })
                          .collect::<Vec<&clean::TraitMethod>>();

    if t.items.len() == 0 {
        try!(write!(w, "{{ }}"));
    } else {
        try!(write!(w, "{{\n"));
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
                    *m.item().name.get_ref(),
                    ConciseStability(&m.item().stability)));
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
            try!(trait_item(w, *m));
        }
        try!(write!(w, "</div>"));
    }
    if provided.len() > 0 {
        try!(write!(w, "
            <h2 id='provided-methods'>Provided Methods</h2>
            <div class='methods'>
        "));
        for m in provided.iter() {
            try!(trait_item(w, *m));
        }
        try!(write!(w, "</div>"));
    }

    let cache = cache_key.get().unwrap();
    try!(write!(w, "
        <h2 id='implementors'>Implementors</h2>
        <ul class='item-list' id='implementors-list'>
    "));
    match cache.implementors.find(&it.def_id) {
        Some(implementors) => {
            for i in implementors.iter() {
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
                root_path = Vec::from_elem(cx.current.len(), "..").connect("/"),
                path = if ast_util::is_local(it.def_id) {
                    cx.current.connect("/")
                } else {
                    let path = cache.external_paths.get(&it.def_id);
                    path.slice_to(path.len() - 1).connect("/")
                },
                ty = shortty(it).to_static_str(),
                name = *it.name.get_ref()));
    Ok(())
}

fn render_method(w: &mut fmt::Formatter, meth: &clean::Item) -> fmt::Result {
    fn fun(w: &mut fmt::Formatter, it: &clean::Item, fn_style: ast::FnStyle,
           g: &clean::Generics, selfty: &clean::SelfTy,
           d: &clean::FnDecl) -> fmt::Result {
        write!(w, "{}fn <a href='#{ty}.{name}' class='fnname'>{name}</a>\
                   {generics}{decl}{where_clause}",
               match fn_style {
                   ast::UnsafeFn => "unsafe ",
                   _ => "",
               },
               ty = shortty(it),
               name = it.name.get_ref().as_slice(),
               generics = *g,
               decl = Method(selfty, d),
               where_clause = WhereClause(g))
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

fn item_struct(w: &mut fmt::Formatter, it: &clean::Item,
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
        doctree::Plain => {
            if fields.peek().is_some() {
                try!(write!(w, "<h2 class='fields'>Fields</h2>\n<table>"));
                for field in fields {
                    try!(write!(w, "<tr><td id='structfield.{name}'>\
                                      {stab}<code>{name}</code></td><td>",
                                  stab = ConciseStability(&field.stability),
                                  name = field.name.get_ref().as_slice()));
                    try!(document(w, field));
                    try!(write!(w, "</td></tr>"));
                }
                try!(write!(w, "</table>"));
            }
        }
        _ => {}
    }
    render_methods(w, it)
}

fn item_enum(w: &mut fmt::Formatter, it: &clean::Item,
             e: &clean::Enum) -> fmt::Result {
    try!(write!(w, "<pre class='rust enum'>{}enum {}{}{}",
                  VisSpace(it.visibility),
                  it.name.get_ref().as_slice(),
                  e.generics,
                  WhereClause(&e.generics)));
    if e.variants.len() == 0 && !e.variants_stripped {
        try!(write!(w, " {{}}"));
    } else {
        try!(write!(w, " {{\n"));
        for v in e.variants.iter() {
            try!(write!(w, "    "));
            let name = v.name.as_ref().unwrap().as_slice();
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
        try!(write!(w, "}}"));
    }
    try!(write!(w, "</pre>"));

    try!(document(w, it));
    if e.variants.len() > 0 {
        try!(write!(w, "<h2 class='variants'>Variants</h2>\n<table>"));
        for variant in e.variants.iter() {
            try!(write!(w, "<tr><td id='variant.{name}'>{stab}<code>{name}</code></td><td>",
                          stab = ConciseStability(&variant.stability),
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

fn render_struct(w: &mut fmt::Formatter, it: &clean::Item,
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
        Some(g) => try!(write!(w, "{}{}", *g, WhereClause(g))),
        None => {}
    }
    match ty {
        doctree::Plain => {
            try!(write!(w, " {{\n{}", tab));
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
    match cache_key.get().unwrap().impls.find(&it.def_id) {
        Some(v) => {
            let (non_trait, traits) = v.partitioned(|i| i.impl_.trait_.is_none());
            if non_trait.len() > 0 {
                try!(write!(w, "<h2 id='methods'>Methods</h2>"));
                for i in non_trait.iter() {
                    try!(render_impl(w, i));
                }
            }
            if traits.len() > 0 {
                try!(write!(w, "<h2 id='implementations'>Trait \
                                  Implementations</h2>"));
                let (derived, manual) = traits.partition(|i| i.impl_.derived);
                for i in manual.iter() {
                    try!(render_impl(w, i));
                }
                if derived.len() > 0 {
                    try!(write!(w, "<h3 id='derived_implementations'>Derived Implementations \
                                </h3>"));
                    for i in derived.iter() {
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
    match i.impl_.trait_ {
        Some(ref ty) => try!(write!(w, "{} for ", *ty)),
        None => {}
    }
    try!(write!(w, "{}{}</code></h3>", i.impl_.for_, WhereClause(&i.impl_.generics)));
    match i.dox {
        Some(ref dox) => {
            try!(write!(w, "<div class='docblock'>{}</div>",
                          Markdown(dox.as_slice())));
        }
        None => {}
    }

    fn doctraititem(w: &mut fmt::Formatter, item: &clean::Item, dox: bool)
                    -> fmt::Result {
        try!(write!(w, "<h4 id='method.{}' class='method'>{}<code>",
                    *item.name.get_ref(),
                    ConciseStability(&item.stability)));
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

    try!(write!(w, "<div class='impl-methods'>"));
    for trait_item in i.impl_.items.iter() {
        try!(doctraititem(w, trait_item, true));
    }

    fn render_default_methods(w: &mut fmt::Formatter,
                              t: &clean::Trait,
                              i: &clean::Impl) -> fmt::Result {
        for trait_item in t.items.iter() {
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
    match i.impl_.trait_ {
        Some(clean::ResolvedPath { did, .. }) => {
            try!({
                match cache_key.get().unwrap().traits.find(&did) {
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
                  it.name.get_ref().as_slice(),
                  t.generics,
                  t.type_));

    document(w, it)
}

impl<'a> fmt::Show for Sidebar<'a> {
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
                          cx.root_path
                            .as_slice()
                            .slice_to((cx.current.len() - i - 1) * 3),
                          *name));
        }
        try!(write!(fmt, "</p>"));

        fn block(w: &mut fmt::Formatter, short: &str, longty: &str,
                 cur: &clean::Item, cx: &Context) -> fmt::Result {
            let items = match cx.sidebar.find_equiv(&short) {
                Some(items) => items.as_slice(),
                None => return Ok(())
            };
            try!(write!(w, "<div class='block {}'><h2>{}</h2>", short, longty));
            for item in items.iter() {
                let curty = shortty(cur).to_static_str();
                let class = if cur.name.as_ref().unwrap() == item &&
                               short == curty { "current" } else { "" };
                try!(write!(w, "<a class='{ty} {class}' href='{href}{path}'>\
                                {name}</a>",
                       ty = short,
                       class = class,
                       href = if curty == "mod" {"../"} else {""},
                       path = if short == "mod" {
                           format!("{}/index.html", item.as_slice())
                       } else {
                           format!("{}.{}.html", short, item.as_slice())
                       },
                       name = item.as_slice()));
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

fn build_sidebar(m: &clean::Module) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::new();
    for item in m.items.iter() {
        if ignore_private_item(item) { continue }

        let short = shortty(item).to_static_str();
        let myname = match item.name {
            None => continue,
            Some(ref s) => s.to_string(),
        };
        let v = match map.entry(short.to_string()) {
            Vacant(entry) => entry.set(Vec::with_capacity(1)),
            Occupied(entry) => entry.into_mut(),
        };
        v.push(myname);
    }

    for (_, items) in map.iter_mut() {
        items.as_mut_slice().sort();
    }
    return map;
}

impl<'a> fmt::Show for Source<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Source(s) = *self;
        let lines = s.lines().count();
        let mut cols = 0;
        let mut tmp = lines;
        while tmp > 0 {
            cols += 1;
            tmp /= 10;
        }
        try!(write!(fmt, "<pre class='line-numbers'>"));
        for i in range(1, lines + 1) {
            try!(write!(fmt, "<span id='{0:u}'>{0:1$u}</span>\n", i, cols));
        }
        try!(write!(fmt, "</pre>"));
        try!(write!(fmt, "{}", highlight::highlight(s.as_slice(), None, None)));
        Ok(())
    }
}

fn item_macro(w: &mut fmt::Formatter, it: &clean::Item,
              t: &clean::Macro) -> fmt::Result {
    try!(w.write(highlight::highlight(t.source.as_slice(), Some("macro"),
                                      None).as_bytes()));
    document(w, it)
}

fn item_primitive(w: &mut fmt::Formatter,
                  it: &clean::Item,
                  _p: &clean::PrimitiveType) -> fmt::Result {
    try!(document(w, it));
    render_methods(w, it)
}

fn ignore_private_item(it: &clean::Item) -> bool {
    match it.inner {
        clean::ModuleItem(ref m) => {
            (m.items.len() == 0 && it.doc_value().is_none()) ||
               it.visibility != Some(ast::Public)
        }
        clean::PrimitiveItem(..) => it.visibility != Some(ast::Public),
        _ => false,
    }
}

fn get_basic_keywords() -> &'static str {
    "rust, rustlang, rust-lang"
}

fn make_item_keywords(it: &clean::Item) -> String {
    format!("{}, {}", get_basic_keywords(), it.name.get_ref())
}
