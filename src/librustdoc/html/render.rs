// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
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
//! and then it is shared among the various rendering threads. The cache is meant
//! to be a fairly large structure not implementing `Clone` (because it's shared
//! among threads). The context, however, should be a lightweight structure. This
//! is cloned per-thread and contains information about what is currently being
//! rendered.
//!
//! In order to speed up rendering (mostly because of markdown rendering), the
//! rendering process has been parallelized. This parallelization is only
//! exposed through the `crate` method on the context, and then also from the
//! fact that the shared cache is stored in TLS (and must be accessed as such).
//!
//! In addition to rendering the crate itself, this module is also responsible
//! for creating the corresponding search index and source file renderings.
//! These threads are not parallelized (they haven't been a bottleneck yet), and
//! both occur before the crate is rendered.
pub use self::ExternalLocation::*;

use std::ascii::AsciiExt;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::default::Default;
use std::error;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufWriter, BufReader};
use std::iter::repeat;
use std::mem;
use std::path::{PathBuf, Path};
use std::str;
use std::sync::Arc;

use externalfiles::ExternalHtml;

use serialize::json::{self, ToJson};
use syntax::{abi, ast};
use rustc::middle::cstore::LOCAL_CRATE;
use rustc::middle::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::middle::privacy::AccessLevels;
use rustc::middle::stability;
use rustc_front::hir;

use clean::{self, SelfTy};
use doctree;
use fold::DocFolder;
use html::escape::Escape;
use html::format::{ConstnessSpace};
use html::format::{TyParamBounds, WhereClause, href, AbiSpace};
use html::format::{VisSpace, Method, UnsafetySpace, MutableSpace};
use html::item_type::ItemType;
use html::markdown::{self, Markdown};
use html::{highlight, layout};

/// A pair of name and its optional document.
pub type NameDoc = (String, Option<String>);

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
    pub src_root: PathBuf,
    /// The current destination folder of where HTML artifacts should be placed.
    /// This changes as the context descends into the module hierarchy.
    pub dst: PathBuf,
    /// This describes the layout of each page, and is not modified after
    /// creation of the context (contains info like the favicon and added html).
    pub layout: layout::Layout,
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
    /// The base-URL of the issue tracker for when an item has been tagged with
    /// an issue number.
    pub issue_tracker_base_url: Option<String>,
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
    pub def_id: DefId,
    pub stability: Option<clean::Stability>,
    pub impl_: clean::Impl,
}

/// Metadata about implementations for a type.
#[derive(Clone)]
pub struct Impl {
    pub impl_: clean::Impl,
    pub dox: Option<String>,
    pub stability: Option<clean::Stability>,
}

impl Impl {
    fn trait_did(&self) -> Option<DefId> {
        self.impl_.trait_.as_ref().and_then(|tr| {
            if let clean::ResolvedPath { did, .. } = *tr {Some(did)} else {None}
        })
    }
}

#[derive(Debug)]
pub struct Error {
    file: PathBuf,
    error: io::Error,
}

impl error::Error for Error {
    fn description(&self) -> &str {
        self.error.description()
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "\"{}\": {}", self.file.display(), self.error)
    }
}

impl Error {
    pub fn new(e: io::Error, file: &Path) -> Error {
        Error {
            file: file.to_path_buf(),
            error: e,
        }
    }
}

macro_rules! try_err {
    ($e:expr, $file:expr) => ({
        match $e {
            Ok(e) => e,
            Err(e) => return Err(Error::new(e, $file)),
        }
    })
}

/// This cache is used to store information about the `clean::Crate` being
/// rendered in order to provide more useful documentation. This contains
/// information like all implementors of a trait, all traits a type implements,
/// documentation for all known traits, etc.
///
/// This structure purposefully does not implement `Clone` because it's intended
/// to be a fairly large and expensive structure to clone. Instead this adheres
/// to `Send` so it may be stored in a `Arc` instance and shared among the various
/// rendering threads.
#[derive(Default)]
pub struct Cache {
    /// Mapping of typaram ids to the name of the type parameter. This is used
    /// when pretty-printing a type (so pretty printing doesn't have to
    /// painfully maintain a context like this)
    pub typarams: HashMap<DefId, String>,

    /// Maps a type id to all known implementations for that type. This is only
    /// recognized for intra-crate `ResolvedPath` types, and is used to print
    /// out extra documentation on the page of an enum/struct.
    ///
    /// The values of the map are a list of implementations and documentation
    /// found on that implementation.
    pub impls: HashMap<DefId, Vec<Impl>>,

    /// Maintains a mapping of local crate node ids to the fully qualified name
    /// and "short type description" of that node. This is used when generating
    /// URLs when a type is being linked to. External paths are not located in
    /// this map because the `External` type itself has all the information
    /// necessary.
    pub paths: HashMap<DefId, (Vec<String>, ItemType)>,

    /// Similar to `paths`, but only holds external paths. This is only used for
    /// generating explicit hyperlinks to other crates.
    pub external_paths: HashMap<DefId, Vec<String>>,

    /// This map contains information about all known traits of this crate.
    /// Implementations of a crate should inherit the documentation of the
    /// parent trait if no extra documentation is specified, and default methods
    /// should show up in documentation about trait implementations.
    pub traits: HashMap<DefId, clean::Trait>,

    /// When rendering traits, it's often useful to be able to list all
    /// implementors of the trait, and this mapping is exactly, that: a mapping
    /// of trait ids to the list of known implementors of the trait
    pub implementors: HashMap<DefId, Vec<Implementor>>,

    /// Cache of where external crate documentation can be found.
    pub extern_locations: HashMap<ast::CrateNum, (String, ExternalLocation)>,

    /// Cache of where documentation for primitives can be found.
    pub primitive_locations: HashMap<clean::PrimitiveType, ast::CrateNum>,

    /// Set of definitions which have been inlined from external crates.
    pub inlined: HashSet<DefId>,

    // Private fields only used when initially crawling a crate to build a cache

    stack: Vec<String>,
    parent_stack: Vec<DefId>,
    search_index: Vec<IndexItem>,
    privmod: bool,
    remove_priv: bool,
    access_levels: AccessLevels<DefId>,
    deref_trait_did: Option<DefId>,

    // In rare case where a structure is defined in one module but implemented
    // in another, if the implementing module is parsed before defining module,
    // then the fully qualified name of the structure isn't presented in `paths`
    // yet when its implementation methods are being indexed. Caches such methods
    // and their parent id here and indexes them at the end of crate parsing.
    orphan_methods: Vec<(DefId, clean::Item)>,
}

/// Helper struct to render all source code to HTML pages
struct SourceCollector<'a> {
    cx: &'a mut Context,

    /// Processed source-file paths
    seen: HashSet<String>,
    /// Root destination to place all HTML output into
    dst: PathBuf,
}

/// Wrapper struct to render the source code of a file. This will do things like
/// adding line numbers to the left-hand side.
struct Source<'a>(&'a str);

// Helper structs for rendering items/sidebars and carrying along contextual
// information

#[derive(Copy, Clone)]
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
    parent: Option<DefId>,
    search_type: Option<IndexItemFunctionType>,
}

/// A type used for the search index.
struct Type {
    name: Option<String>,
}

impl fmt::Display for Type {
    /// Formats type as {name: $name}.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Wrapping struct fmt should never call us when self.name is None,
        // but just to be safe we write `null` in that case.
        match self.name {
            Some(ref n) => write!(f, "{{\"name\":\"{}\"}}", n),
            None => write!(f, "null")
        }
    }
}

/// Full type of functions/methods in the search index.
struct IndexItemFunctionType {
    inputs: Vec<Type>,
    output: Option<Type>
}

impl fmt::Display for IndexItemFunctionType {
    /// Formats a full fn type as a JSON {inputs: [Type], outputs: Type/null}.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // If we couldn't figure out a type, just write `null`.
        if self.inputs.iter().any(|ref i| i.name.is_none()) ||
           (self.output.is_some() && self.output.as_ref().unwrap().name.is_none()) {
            return write!(f, "null")
        }

        let inputs: Vec<String> = self.inputs.iter().map(|ref t| {
            format!("{}", t)
        }).collect();
        try!(write!(f, "{{\"inputs\":[{}],\"output\":", inputs.join(",")));

        match self.output {
            Some(ref t) => try!(write!(f, "{}", t)),
            None => try!(write!(f, "null"))
        };

        Ok(try!(write!(f, "}}")))
    }
}

// TLS keys used to carry information around during rendering.

thread_local!(static CACHE_KEY: RefCell<Arc<Cache>> = Default::default());
thread_local!(pub static CURRENT_LOCATION_KEY: RefCell<Vec<String>> =
                    RefCell::new(Vec::new()));
thread_local!(static USED_ID_MAP: RefCell<HashMap<String, usize>> =
                    RefCell::new(HashMap::new()));

/// This method resets the local table of used ID attributes. This is typically
/// used at the beginning of rendering an entire HTML page to reset from the
/// previous state (if any).
pub fn reset_ids() {
    USED_ID_MAP.with(|s| s.borrow_mut().clear());
}

pub fn with_unique_id<T, F: FnOnce(&str) -> T>(candidate: String, f: F) -> T {
    USED_ID_MAP.with(|map| {
        let (id, ret) = match map.borrow_mut().get_mut(&candidate) {
            None => {
                let ret = f(&candidate);
                (candidate, ret)
            },
            Some(a) => {
                let id = format!("{}-{}", candidate, *a);
                let ret = f(&id);
                *a += 1;
                (id, ret)
            }
        };

        map.borrow_mut().insert(id, 1);
        ret
    })
}

/// Generates the documentation for `crate` into the directory `dst`
pub fn run(mut krate: clean::Crate,
           external_html: &ExternalHtml,
           dst: PathBuf,
           passes: HashSet<String>) -> Result<(), Error> {
    let src_root = match krate.src.parent() {
        Some(p) => p.to_path_buf(),
        None => PathBuf::new(),
    };
    let mut cx = Context {
        dst: dst,
        src_root: src_root,
        passes: passes,
        current: Vec::new(),
        root_path: String::new(),
        layout: layout::Layout {
            logo: "".to_string(),
            favicon: "".to_string(),
            external_html: external_html.clone(),
            krate: krate.name.clone(),
            playground_url: "".to_string(),
        },
        include_sources: true,
        render_redirect_pages: false,
        issue_tracker_base_url: None,
    };

    try_err!(mkdir(&cx.dst), &cx.dst);

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
                    clean::NameValue(ref x, ref s)
                            if "issue_tracker_base_url" == *x => {
                        cx.issue_tracker_base_url = Some(s.to_string());
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
    let access_levels = analysis.as_ref().map(|a| a.access_levels.clone());
    let access_levels = access_levels.unwrap_or(Default::default());
    let paths: HashMap<DefId, (Vec<String>, ItemType)> =
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
        access_levels: access_levels,
        orphan_methods: Vec::new(),
        traits: mem::replace(&mut krate.external_traits, HashMap::new()),
        deref_trait_did: analysis.as_ref().and_then(|a| a.deref_trait_did),
        typarams: analysis.as_ref().map(|a| {
            a.external_typarams.borrow_mut().take().unwrap()
        }).unwrap_or(HashMap::new()),
        inlined: analysis.as_ref().map(|a| {
            a.inlined.borrow_mut().take().unwrap()
        }).unwrap_or(HashSet::new()),
    };

    // Cache where all our extern crates are located
    for &(n, ref e) in &krate.externs {
        cache.extern_locations.insert(n, (e.name.clone(),
                                          extern_location(e, &cx.dst)));
        let did = DefId { krate: n, index: CRATE_DEF_INDEX };
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
        cache.primitive_locations.insert(prim, LOCAL_CRATE);
    }

    cache.stack.push(krate.name.clone());
    krate = cache.fold_crate(krate);

    // Build our search index
    let index = build_index(&krate, &mut cache);

    // Freeze the cache now that the index has been built. Put an Arc into TLS
    // for future parallelization opportunities
    let cache = Arc::new(cache);
    CACHE_KEY.with(|v| *v.borrow_mut() = cache.clone());
    CURRENT_LOCATION_KEY.with(|s| s.borrow_mut().clear());

    try!(write_shared(&cx, &krate, &*cache, index));
    let krate = try!(render_sources(&mut cx, krate));

    // And finally render the whole crate's documentation
    cx.krate(krate)
}

fn build_index(krate: &clean::Crate, cache: &mut Cache) -> String {
    // Build the search index from the collected metadata
    let mut nodeid_to_pathid = HashMap::new();
    let mut pathid_to_nodeid = Vec::new();
    {
        let Cache { ref mut search_index,
                    ref orphan_methods,
                    ref mut paths, .. } = *cache;

        // Attach all orphan methods to the type's definition if the type
        // has since been learned.
        for &(did, ref item) in orphan_methods {
            match paths.get(&did) {
                Some(&(ref fqp, _)) => {
                    // Needed to determine `self` type.
                    let parent_basename = Some(fqp[fqp.len() - 1].clone());
                    search_index.push(IndexItem {
                        ty: shortty(item),
                        name: item.name.clone().unwrap(),
                        path: fqp[..fqp.len() - 1].join("::"),
                        desc: shorter(item.doc_value()),
                        parent: Some(did),
                        search_type: get_index_search_type(&item, parent_basename),
                    });
                },
                None => {}
            }
        }

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
    let mut w = io::Cursor::new(Vec::new());
    write!(&mut w, r#"searchIndex['{}'] = {{"items":["#, krate.name).unwrap();

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
            write!(&mut w, ",").unwrap();
        }
        write!(&mut w, r#"[{},"{}","{}",{}"#,
               item.ty as usize, item.name, path,
               item.desc.to_json().to_string()).unwrap();
        match item.parent {
            Some(nodeid) => {
                let pathid = *nodeid_to_pathid.get(&nodeid).unwrap();
                write!(&mut w, ",{}", pathid).unwrap();
            }
            None => write!(&mut w, ",null").unwrap()
        }
        match item.search_type {
            Some(ref t) => write!(&mut w, ",{}", t).unwrap(),
            None => write!(&mut w, ",null").unwrap()
        }
        write!(&mut w, "]").unwrap();
    }

    write!(&mut w, r#"],"paths":["#).unwrap();

    for (i, &did) in pathid_to_nodeid.iter().enumerate() {
        let &(ref fqp, short) = cache.paths.get(&did).unwrap();
        if i > 0 {
            write!(&mut w, ",").unwrap();
        }
        write!(&mut w, r#"[{},"{}"]"#,
               short as usize, *fqp.last().unwrap()).unwrap();
    }

    write!(&mut w, "]}};").unwrap();

    String::from_utf8(w.into_inner()).unwrap()
}

fn write_shared(cx: &Context,
                krate: &clean::Crate,
                cache: &Cache,
                search_index: String) -> Result<(), Error> {
    // Write out the shared files. Note that these are shared among all rustdoc
    // docs placed in the output directory, so this needs to be a synchronized
    // operation with respect to all other rustdocs running around.
    try_err!(mkdir(&cx.dst), &cx.dst);
    let _lock = ::flock::Lock::new(&cx.dst.join(".lock"));

    // Add all the static files. These may already exist, but we just
    // overwrite them anyway to make sure that they're fresh and up-to-date.
    try!(write(cx.dst.join("jquery.js"),
               include_bytes!("static/jquery-2.1.4.min.js")));
    try!(write(cx.dst.join("main.js"),
               include_bytes!("static/main.js")));
    try!(write(cx.dst.join("playpen.js"),
               include_bytes!("static/playpen.js")));
    try!(write(cx.dst.join("main.css"),
               include_bytes!("static/main.css")));
    try!(write(cx.dst.join("normalize.css"),
               include_bytes!("static/normalize.css")));
    try!(write(cx.dst.join("FiraSans-Regular.woff"),
               include_bytes!("static/FiraSans-Regular.woff")));
    try!(write(cx.dst.join("FiraSans-Medium.woff"),
               include_bytes!("static/FiraSans-Medium.woff")));
    try!(write(cx.dst.join("FiraSans-LICENSE.txt"),
               include_bytes!("static/FiraSans-LICENSE.txt")));
    try!(write(cx.dst.join("Heuristica-Italic.woff"),
               include_bytes!("static/Heuristica-Italic.woff")));
    try!(write(cx.dst.join("Heuristica-LICENSE.txt"),
               include_bytes!("static/Heuristica-LICENSE.txt")));
    try!(write(cx.dst.join("SourceSerifPro-Regular.woff"),
               include_bytes!("static/SourceSerifPro-Regular.woff")));
    try!(write(cx.dst.join("SourceSerifPro-Bold.woff"),
               include_bytes!("static/SourceSerifPro-Bold.woff")));
    try!(write(cx.dst.join("SourceSerifPro-LICENSE.txt"),
               include_bytes!("static/SourceSerifPro-LICENSE.txt")));
    try!(write(cx.dst.join("SourceCodePro-Regular.woff"),
               include_bytes!("static/SourceCodePro-Regular.woff")));
    try!(write(cx.dst.join("SourceCodePro-Semibold.woff"),
               include_bytes!("static/SourceCodePro-Semibold.woff")));
    try!(write(cx.dst.join("SourceCodePro-LICENSE.txt"),
               include_bytes!("static/SourceCodePro-LICENSE.txt")));
    try!(write(cx.dst.join("LICENSE-MIT.txt"),
               include_bytes!("static/LICENSE-MIT.txt")));
    try!(write(cx.dst.join("LICENSE-APACHE.txt"),
               include_bytes!("static/LICENSE-APACHE.txt")));
    try!(write(cx.dst.join("COPYRIGHT.txt"),
               include_bytes!("static/COPYRIGHT.txt")));

    fn collect(path: &Path, krate: &str,
               key: &str) -> io::Result<Vec<String>> {
        let mut ret = Vec::new();
        if path.exists() {
            for line in BufReader::new(try!(File::open(path))).lines() {
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
    let all_indexes = try_err!(collect(&dst, &krate.name, "searchIndex"), &dst);
    let mut w = try_err!(File::create(&dst), &dst);
    try_err!(writeln!(&mut w, "var searchIndex = {{}};"), &dst);
    try_err!(writeln!(&mut w, "{}", search_index), &dst);
    for index in &all_indexes {
        try_err!(writeln!(&mut w, "{}", *index), &dst);
    }
    try_err!(writeln!(&mut w, "initSearch(searchIndex);"), &dst);

    // Update the list of all implementors for traits
    let dst = cx.dst.join("implementors");
    try_err!(mkdir(&dst), &dst);
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
            try_err!(mkdir(&mydst), &mydst);
        }
        mydst.push(&format!("{}.{}.js",
                            remote_item_type.to_static_str(),
                            remote_path[remote_path.len() - 1]));
        let all_implementors = try_err!(collect(&mydst, &krate.name,
                                                "implementors"),
                                        &mydst);

        try_err!(mkdir(mydst.parent().unwrap()),
                 &mydst.parent().unwrap().to_path_buf());
        let mut f = BufWriter::new(try_err!(File::create(&mydst), &mydst));
        try_err!(writeln!(&mut f, "(function() {{var implementors = {{}};"), &mydst);

        for implementor in &all_implementors {
            try_err!(write!(&mut f, "{}", *implementor), &mydst);
        }

        try_err!(write!(&mut f, r"implementors['{}'] = [", krate.name), &mydst);
        for imp in imps {
            // If the trait and implementation are in the same crate, then
            // there's no need to emit information about it (there's inlining
            // going on). If they're in different crates then the crate defining
            // the trait will be interested in our implementation.
            if imp.def_id.krate == did.krate { continue }
            try_err!(write!(&mut f, r#""{}","#, imp.impl_), &mydst);
        }
        try_err!(writeln!(&mut f, r"];"), &mydst);
        try_err!(writeln!(&mut f, "{}", r"
            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        "), &mydst);
        try_err!(writeln!(&mut f, r"}})()"), &mydst);
    }
    Ok(())
}

fn render_sources(cx: &mut Context,
                  krate: clean::Crate) -> Result<clean::Crate, Error> {
    info!("emitting source files");
    let dst = cx.dst.join("src");
    try_err!(mkdir(&dst), &dst);
    let dst = dst.join(&krate.name);
    try_err!(mkdir(&dst), &dst);
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
fn write(dst: PathBuf, contents: &[u8]) -> Result<(), Error> {
    Ok(try_err!(try_err!(File::create(&dst), &dst).write_all(contents), &dst))
}

/// Makes a directory on the filesystem, failing the thread if an error occurs and
/// skipping if the directory already exists.
fn mkdir(path: &Path) -> io::Result<()> {
    if !path.exists() {
        fs::create_dir(path)
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
/// static HTML tree. Each component in the cleaned path will be passed as an
/// argument to `f`. The very last component of the path (ie the file name) will
/// be passed to `f` if `keep_filename` is true, and ignored otherwise.
// FIXME (#9639): The closure should deal with &[u8] instead of &str
// FIXME (#9639): This is too conservative, rejecting non-UTF-8 paths
fn clean_srcpath<F>(src_root: &Path, p: &Path, keep_filename: bool, mut f: F) where
    F: FnMut(&str),
{
    // make it relative, if possible
    let p = p.relative_from(src_root).unwrap_or(p);

    let mut iter = p.iter().map(|x| x.to_str().unwrap()).peekable();
    while let Some(c) = iter.next() {
        if !keep_filename && iter.peek().is_none() {
            break;
        }

        if ".." == c {
            f("up");
        } else {
            f(c)
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
    fn emit_source(&mut self, filename: &str) -> io::Result<()> {
        let p = PathBuf::from(filename);

        // If we couldn't open this file, then just returns because it
        // probably means that it's some standard library macro thing and we
        // can't have the source to it anyway.
        let mut contents = Vec::new();
        match File::open(&p).and_then(|mut f| f.read_to_end(&mut contents)) {
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
        let mut root_path = String::from("../../");
        clean_srcpath(&self.cx.src_root, &p, false, |component| {
            cur.push(component);
            mkdir(&cur).unwrap();
            root_path.push_str("../");
        });

        let mut fname = p.file_name().expect("source has no filename")
                         .to_os_string();
        fname.push(".html");
        cur.push(&fname[..]);
        let mut w = BufWriter::new(try!(File::create(&cur)));
        let title = format!("{} -- source", cur.file_name().unwrap()
                                               .to_string_lossy());
        let desc = format!("Source to the Rust file `{}`.", filename);
        let page = layout::Page {
            title: &title,
            ty: "source",
            root_path: &root_path,
            description: &desc,
            keywords: get_basic_keywords(),
        };
        try!(layout::render(&mut w, &self.cx.layout,
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
                self.privmod = prev || (self.remove_priv && item.visibility != Some(hir::Public));
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
            clean::TypedefItem(ref t, _)      => self.generics(&t.generics),
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
                    self.implementors.entry(did).or_insert(vec![]).push(Implementor {
                        def_id: item.def_id,
                        stability: item.stability.clone(),
                        impl_: i.clone(),
                    });
                }
                Some(..) | None => {}
            }
        }

        // Index this method for searching later on
        if let Some(ref s) = item.name {
            let (parent, is_method) = match item.inner {
                clean::AssociatedTypeItem(..) |
                clean::AssociatedConstItem(..) |
                clean::TyMethodItem(..) |
                clean::StructFieldItem(..) |
                clean::VariantItem(..) => {
                    ((Some(*self.parent_stack.last().unwrap()),
                      Some(&self.stack[..self.stack.len() - 1])),
                     false)
                }
                clean::MethodItem(..) => {
                    if self.parent_stack.is_empty() {
                        ((None, None), false)
                    } else {
                        let last = self.parent_stack.last().unwrap();
                        let did = *last;
                        let path = match self.paths.get(&did) {
                            Some(&(_, ItemType::Trait)) =>
                                Some(&self.stack[..self.stack.len() - 1]),
                            // The current stack not necessarily has correlation
                            // for where the type was defined. On the other
                            // hand, `paths` always has the right
                            // information if present.
                            Some(&(ref fqp, ItemType::Struct)) |
                            Some(&(ref fqp, ItemType::Enum)) =>
                                Some(&fqp[..fqp.len() - 1]),
                            Some(..) => Some(&*self.stack),
                            None => None
                        };
                        ((Some(*last), path), true)
                    }
                }
                clean::TypedefItem(_, true) => {
                    // skip associated types in impls
                    ((None, None), false)
                }
                _ => ((None, Some(&*self.stack)), false)
            };
            let hidden_field = match item.inner {
                clean::StructFieldItem(clean::HiddenStructField) => true,
                _ => false
            };

            match parent {
                (parent, Some(path)) if is_method || (!self.privmod && !hidden_field) => {
                    // Needed to determine `self` type.
                    let parent_basename = self.parent_stack.first().and_then(|parent| {
                        match self.paths.get(parent) {
                            Some(&(ref fqp, _)) => Some(fqp[fqp.len() - 1].clone()),
                            _ => None
                        }
                    });

                    self.search_index.push(IndexItem {
                        ty: shortty(&item),
                        name: s.to_string(),
                        path: path.join("::").to_string(),
                        desc: shorter(item.doc_value()),
                        parent: parent,
                        search_type: get_index_search_type(&item, parent_basename),
                    });
                }
                (Some(parent), None) if is_method || (!self.privmod && !hidden_field)=> {
                    if parent.is_local() {
                        // We have a parent, but we don't know where they're
                        // defined yet. Wait for later to index this item.
                        self.orphan_methods.push((parent, item.clone()))
                    }
                }
                _ => {}
            }
        }

        // Keep track of the fully qualified path for this item.
        let pushed = if item.name.is_some() {
            let n = item.name.as_ref().unwrap();
            if !n.is_empty() {
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
                if
                    !self.paths.contains_key(&item.def_id) ||
                    !item.def_id.is_local() ||
                    self.access_levels.is_public(item.def_id)
                {
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
                    ref t => {
                        match t.primitive_type() {
                            Some(prim) => {
                                let did = DefId::local(prim.to_def_index());
                                self.parent_stack.push(did);
                                true
                            }
                            _ => false,
                        }
                    }
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
                            clean::ResolvedPath { did, .. } |
                            clean::BorrowedRef {
                                type_: box clean::ResolvedPath { did, .. }, ..
                            } => {
                                Some(did)
                            }

                            ref t => {
                                t.primitive_type().and_then(|t| {
                                    self.primitive_locations.get(&t).map(|n| {
                                        let id = t.to_def_index();
                                        DefId { krate: *n, index: id }
                                    })
                                })
                            }
                        };

                        if let Some(did) = did {
                            self.impls.entry(did).or_insert(vec![]).push(Impl {
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
        if s.is_empty() {
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
    fn krate(self, mut krate: clean::Crate) -> Result<(), Error> {
        let mut item = match krate.module.take() {
            Some(i) => i,
            None => return Ok(())
        };
        item.name = Some(krate.name);

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
    fn item<F>(&mut self, item: clean::Item, mut f: F) -> Result<(), Error> where
        F: FnMut(&mut Context, clean::Item),
    {
        fn render(w: File, cx: &Context, it: &clean::Item,
                  pushname: bool) -> io::Result<()> {
            // A little unfortunate that this is done like this, but it sure
            // does make formatting *a lot* nicer.
            CURRENT_LOCATION_KEY.with(|slot| {
                *slot.borrow_mut() = cx.current.clone();
            });

            let mut title = cx.current.join("::");
            if pushname {
                if !title.is_empty() {
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

            reset_ids();

            // We have a huge number of calls to write, so try to alleviate some
            // of the pain by using a buffered writer instead of invoking the
            // write syscall all the time.
            let mut writer = BufWriter::new(w);
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
                    let joint_dst = this.dst.join("index.html");
                    let dst = try_err!(File::create(&joint_dst), &joint_dst);
                    try_err!(render(dst, this, &item, false), &joint_dst);

                    let m = match item.inner {
                        clean::ModuleItem(m) => m,
                        _ => unreachable!()
                    };

                    // render sidebar-items.js used throughout this module
                    {
                        let items = this.build_sidebar_items(&m);
                        let js_dst = this.dst.join("sidebar-items.js");
                        let mut js_out = BufWriter::new(try_err!(File::create(&js_dst), &js_dst));
                        try_err!(write!(&mut js_out, "initSidebarItems({});",
                                    json::as_json(&items)), &js_dst);
                    }

                    for item in m.items {
                        f(this,item);
                    }
                    Ok(())
                })
            }

            // Things which don't have names (like impls) don't get special
            // pages dedicated to them.
            _ if item.name.is_some() => {
                let joint_dst = self.dst.join(&item_path(&item));

                let dst = try_err!(File::create(&joint_dst), &joint_dst);
                try_err!(render(dst, self, &item, true), &joint_dst);
                Ok(())
            }

            _ => Ok(())
        }
    }

    fn build_sidebar_items(&self, m: &clean::Module) -> BTreeMap<String, Vec<NameDoc>> {
        // BTreeMap instead of HashMap to get a sorted output
        let mut map = BTreeMap::new();
        for item in &m.items {
            if self.ignore_private_item(item) { continue }

            let short = shortty(item).to_static_str();
            let myname = match item.name {
                None => continue,
                Some(ref s) => s.to_string(),
            };
            let short = short.to_string();
            map.entry(short).or_insert(vec![])
                .push((myname, Some(plain_summary_line(item.doc_value()))));
        }

        for (_, items) in &mut map {
            items.sort();
        }
        return map;
    }

    fn ignore_private_item(&self, it: &clean::Item) -> bool {
        match it.inner {
            clean::ModuleItem(ref m) => {
                (m.items.is_empty() &&
                 it.doc_value().is_none() &&
                 it.visibility != Some(hir::Public)) ||
                (self.passes.contains("strip-private") && it.visibility != Some(hir::Public))
            }
            clean::PrimitiveItem(..) => it.visibility != Some(hir::Public),
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
        let href = if self.item.source.loline == self.item.source.hiline {
            format!("{}", self.item.source.loline)
        } else {
            format!("{}-{}", self.item.source.loline, self.item.source.hiline)
        };

        // First check to see if this is an imported macro source. In this case
        // we need to handle it specially as cross-crate inlined macros have...
        // odd locations!
        let imported_macro_from = match self.item.inner {
            clean::MacroItem(ref m) => m.imported_from.as_ref(),
            _ => None,
        };
        if let Some(krate) = imported_macro_from {
            let cache = cache();
            let root = cache.extern_locations.values().find(|&&(ref n, _)| {
                *krate == *n
            }).map(|l| &l.1);
            let root = match root {
                Some(&Remote(ref s)) => s.to_string(),
                Some(&Local) => self.cx.root_path.clone(),
                None | Some(&Unknown) => return None,
            };
            Some(format!("{root}/{krate}/macro.{name}.html?gotomacrosrc=1",
                         root = root,
                         krate = krate,
                         name = self.item.name.as_ref().unwrap()))

        // If this item is part of the local crate, then we're guaranteed to
        // know the span, so we plow forward and generate a proper url. The url
        // has anchors for the line numbers that we're linking to.
        } else if self.item.def_id.is_local() {
            let mut path = Vec::new();
            clean_srcpath(&cx.src_root, Path::new(&self.item.source.filename),
                          true, |component| {
                path.push(component.to_string());
            });
            Some(format!("{root}src/{krate}/{path}.html#{href}",
                         root = self.cx.root_path,
                         krate = self.cx.layout.krate,
                         path = path.join("/"),
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
            let path = &cache.external_paths[&self.item.def_id];
            let root = match cache.extern_locations[&self.item.def_id.krate] {
                (_, Remote(ref s)) => s.to_string(),
                (_, Local) => self.cx.root_path.clone(),
                (_, Unknown) => return None,
            };
            Some(format!("{root}{path}/{file}?gotosrc={goto}",
                         root = root,
                         path = path[..path.len() - 1].join("/"),
                         file = item_path(self.item),
                         goto = self.item.def_id.index.as_usize()))
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

        try!(write!(fmt, "</span>")); // in-band
        try!(write!(fmt, "<span class='out-of-band'>"));
        try!(write!(fmt,
        r##"<span id='render-detail'>
            <a id="toggle-all-docs" href="javascript:void(0)" title="collapse all docs">
                [<span class='inner'>&#x2212;</span>]
            </a>
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
                    try!(write!(fmt, "<a id='src-{}' class='srclink' \
                                       href='{}' title='{}'>[src]</a>",
                                self.item.def_id.index.as_usize(), l, "goto source code"));
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
                item_function(fmt, self.cx, self.item, f),
            clean::TraitItem(ref t) => item_trait(fmt, self.cx, self.item, t),
            clean::StructItem(ref s) => item_struct(fmt, self.cx, self.item, s),
            clean::EnumItem(ref e) => item_enum(fmt, self.cx, self.item, e),
            clean::TypedefItem(ref t, _) => item_typedef(fmt, self.cx, self.item, t),
            clean::MacroItem(ref m) => item_macro(fmt, self.cx, self.item, m),
            clean::PrimitiveItem(ref p) => item_primitive(fmt, self.cx, self.item, p),
            clean::StaticItem(ref i) | clean::ForeignStaticItem(ref i) =>
                item_static(fmt, self.cx, self.item, i),
            clean::ConstantItem(ref c) => item_constant(fmt, self.cx, self.item, c),
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
    let mut s = cx.current.join("::");
    s.push_str("::");
    s.push_str(item.name.as_ref().unwrap());
    return s
}

fn shorter<'a>(s: Option<&'a str>) -> String {
    match s {
        Some(s) => s.lines().take_while(|line|{
            (*line).chars().any(|chr|{
                !chr.is_whitespace()
            })
        }).collect::<Vec<_>>().join("\n"),
        None => "".to_string()
    }
}

#[inline]
fn plain_summary_line(s: Option<&str>) -> String {
    let line = shorter(s).replace("\n", " ");
    markdown::plain_summary_line(&line[..])
}

fn document(w: &mut fmt::Formatter, cx: &Context, item: &clean::Item) -> fmt::Result {
    if let Some(s) = short_stability(item, cx, true) {
        try!(write!(w, "<div class='stability'>{}</div>", s));
    }
    if let Some(s) = item.doc_value() {
        try!(write!(w, "<div class='docblock'>{}</div>", Markdown(s)));
    }
    Ok(())
}

fn item_module(w: &mut fmt::Formatter, cx: &Context,
               item: &clean::Item, items: &[clean::Item]) -> fmt::Result {
    try!(document(w, cx, item));

    let mut indices = (0..items.len()).filter(|i| {
        !cx.ignore_private_item(&items[*i])
    }).collect::<Vec<usize>>();

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

    fn cmp(i1: &clean::Item, i2: &clean::Item, idx1: usize, idx2: usize) -> Ordering {
        let ty1 = shortty(i1);
        let ty2 = shortty(i2);
        if ty1 != ty2 {
            return (reorder(ty1), idx1).cmp(&(reorder(ty2), idx2))
        }
        let s1 = i1.stability.as_ref().map(|s| s.level);
        let s2 = i2.stability.as_ref().map(|s| s.level);
        match (s1, s2) {
            (Some(stability::Unstable), Some(stability::Stable)) => return Ordering::Greater,
            (Some(stability::Stable), Some(stability::Unstable)) => return Ordering::Less,
            _ => {}
        }
        i1.name.cmp(&i2.name)
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
                ItemType::AssociatedConst => ("associated-consts", "Associated Constants"),
            };
            try!(with_unique_id(short.to_owned(), |id|
                write!(w, "<h2 id='{id}' class='section-header'>\
                          <a href=\"#{id}\">{name}</a></h2>\n<table>",
                          id = id, name = name)));
        }

        match myitem.inner {
            clean::ExternCrateItem(ref name, ref src) => {
                match *src {
                    Some(ref src) => {
                        try!(write!(w, "<tr><td><code>{}extern crate {} as {};",
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
                let stab_docs = if let Some(s) = short_stability(myitem, cx, false) {
                    format!("[{}]", s)
                } else {
                    String::new()
                };
                try!(write!(w, "
                    <tr class='{stab} module-item'>
                        <td><a class='{class}' href='{href}'
                               title='{title}'>{name}</a></td>
                        <td class='docblock short'>
                            {stab_docs} {docs}
                        </td>
                    </tr>
                ",
                name = *myitem.name.as_ref().unwrap(),
                stab_docs = stab_docs,
                docs = Markdown(&shorter(myitem.doc_value())),
                class = shortty(myitem),
                stab = myitem.stability_class(),
                href = item_path(myitem),
                title = full_path(cx, myitem)));
            }
        }
    }

    write!(w, "</table>")
}

fn short_stability(item: &clean::Item, cx: &Context, show_reason: bool) -> Option<String> {
    item.stability.as_ref().and_then(|stab| {
        let reason = if show_reason && !stab.reason.is_empty() {
            format!(": {}", stab.reason)
        } else {
            String::new()
        };
        let text = if !stab.deprecated_since.is_empty() {
            let since = if show_reason {
                format!(" since {}", Escape(&stab.deprecated_since))
            } else {
                String::new()
            };
            format!("Deprecated{}{}", since, Markdown(&reason))
        } else if stab.level == stability::Unstable {
            let unstable_extra = if show_reason {
                match (!stab.feature.is_empty(), &cx.issue_tracker_base_url, stab.issue) {
                    (true, &Some(ref tracker_url), Some(issue_no)) =>
                        format!(" (<code>{}</code> <a href=\"{}{}\">#{}</a>)",
                                Escape(&stab.feature), tracker_url, issue_no, issue_no),
                    (false, &Some(ref tracker_url), Some(issue_no)) =>
                        format!(" (<a href=\"{}{}\">#{}</a>)", Escape(&tracker_url), issue_no,
                                issue_no),
                    (true, _, _) =>
                        format!(" (<code>{}</code>)", Escape(&stab.feature)),
                    _ => String::new(),
                }
            } else {
                String::new()
            };
            format!("Unstable{}{}", unstable_extra, Markdown(&reason))
        } else {
            return None
        };
        Some(format!("<em class='stab {}'>{}</em>",
                     item.stability_class(), text))
    })
}

struct Initializer<'a>(&'a str);

impl<'a> fmt::Display for Initializer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Initializer(s) = *self;
        if s.is_empty() { return Ok(()); }
        try!(write!(f, "<code> = </code>"));
        write!(f, "<code>{}</code>", s)
    }
}

fn item_constant(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
                 c: &clean::Constant) -> fmt::Result {
    try!(write!(w, "<pre class='rust const'>{vis}const \
                    {name}: {typ}{init}</pre>",
           vis = VisSpace(it.visibility),
           name = it.name.as_ref().unwrap(),
           typ = c.type_,
           init = Initializer(&c.expr)));
    document(w, cx, it)
}

fn item_static(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
               s: &clean::Static) -> fmt::Result {
    try!(write!(w, "<pre class='rust static'>{vis}static {mutability}\
                    {name}: {typ}{init}</pre>",
           vis = VisSpace(it.visibility),
           mutability = MutableSpace(s.mutability),
           name = it.name.as_ref().unwrap(),
           typ = s.type_,
           init = Initializer(&s.expr)));
    document(w, cx, it)
}

fn item_function(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
                 f: &clean::Function) -> fmt::Result {
    try!(write!(w, "<pre class='rust fn'>{vis}{constness}{unsafety}{abi}fn \
                    {name}{generics}{decl}{where_clause}</pre>",
           vis = VisSpace(it.visibility),
           constness = ConstnessSpace(f.constness),
           unsafety = UnsafetySpace(f.unsafety),
           abi = AbiSpace(f.abi),
           name = it.name.as_ref().unwrap(),
           generics = f.generics,
           where_clause = WhereClause(&f.generics),
           decl = f.decl));
    document(w, cx, it)
}

fn item_trait(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
              t: &clean::Trait) -> fmt::Result {
    let mut bounds = String::new();
    if !t.bounds.is_empty() {
        if !bounds.is_empty() {
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

    let types = t.items.iter().filter(|m| {
        match m.inner { clean::AssociatedTypeItem(..) => true, _ => false }
    }).collect::<Vec<_>>();
    let consts = t.items.iter().filter(|m| {
        match m.inner { clean::AssociatedConstItem(..) => true, _ => false }
    }).collect::<Vec<_>>();
    let required = t.items.iter().filter(|m| {
        match m.inner { clean::TyMethodItem(_) => true, _ => false }
    }).collect::<Vec<_>>();
    let provided = t.items.iter().filter(|m| {
        match m.inner { clean::MethodItem(_) => true, _ => false }
    }).collect::<Vec<_>>();

    if t.items.is_empty() {
        try!(write!(w, "{{ }}"));
    } else {
        try!(write!(w, "{{\n"));
        for t in &types {
            try!(write!(w, "    "));
            try!(render_assoc_item(w, t, AssocItemLink::Anchor));
            try!(write!(w, ";\n"));
        }
        if !types.is_empty() && !consts.is_empty() {
            try!(w.write_str("\n"));
        }
        for t in &consts {
            try!(write!(w, "    "));
            try!(render_assoc_item(w, t, AssocItemLink::Anchor));
            try!(write!(w, ";\n"));
        }
        if !consts.is_empty() && !required.is_empty() {
            try!(w.write_str("\n"));
        }
        for m in &required {
            try!(write!(w, "    "));
            try!(render_assoc_item(w, m, AssocItemLink::Anchor));
            try!(write!(w, ";\n"));
        }
        if !required.is_empty() && !provided.is_empty() {
            try!(w.write_str("\n"));
        }
        for m in &provided {
            try!(write!(w, "    "));
            try!(render_assoc_item(w, m, AssocItemLink::Anchor));
            try!(write!(w, " {{ ... }}\n"));
        }
        try!(write!(w, "}}"));
    }
    try!(write!(w, "</pre>"));

    // Trait documentation
    try!(document(w, cx, it));

    fn trait_item(w: &mut fmt::Formatter, cx: &Context, m: &clean::Item)
                  -> fmt::Result {
        let name = m.name.as_ref().unwrap();
        try!(with_unique_id(format!("{}.{}", shortty(m), name), |id|
                write!(w, "<h3 id='{id}' class='method stab {stab}'><code>",
                       id = id,
                       stab = m.stability_class())));
        try!(render_assoc_item(w, m, AssocItemLink::Anchor));
        try!(write!(w, "</code></h3>"));
        try!(document(w, cx, m));
        Ok(())
    }

    if !types.is_empty() {
        try!(write!(w, "
            <h2 id='associated-types'>Associated Types</h2>
            <div class='methods'>
        "));
        for t in &types {
            try!(trait_item(w, cx, *t));
        }
        try!(write!(w, "</div>"));
    }

    if !consts.is_empty() {
        try!(write!(w, "
            <h2 id='associated-const'>Associated Constants</h2>
            <div class='methods'>
        "));
        for t in &consts {
            try!(trait_item(w, cx, *t));
        }
        try!(write!(w, "</div>"));
    }

    // Output the documentation for each function individually
    if !required.is_empty() {
        try!(write!(w, "
            <h2 id='required-methods'>Required Methods</h2>
            <div class='methods'>
        "));
        for m in &required {
            try!(trait_item(w, cx, *m));
        }
        try!(write!(w, "</div>"));
    }
    if !provided.is_empty() {
        try!(write!(w, "
            <h2 id='provided-methods'>Provided Methods</h2>
            <div class='methods'>
        "));
        for m in &provided {
            try!(trait_item(w, cx, *m));
        }
        try!(write!(w, "</div>"));
    }

    // If there are methods directly on this trait object, render them here.
    try!(render_assoc_items(w, cx, it.def_id, AssocItemRender::All));

    let cache = cache();
    try!(write!(w, "
        <h2 id='implementors'>Implementors</h2>
        <ul class='item-list' id='implementors-list'>
    "));
    match cache.implementors.get(&it.def_id) {
        Some(implementors) => {
            for i in implementors {
                try!(writeln!(w, "<li><code>{}</code></li>", i.impl_));
            }
        }
        None => {}
    }
    try!(write!(w, "</ul>"));
    try!(write!(w, r#"<script type="text/javascript" async
                              src="{root_path}/implementors/{path}/{ty}.{name}.js">
                      </script>"#,
                root_path = vec![".."; cx.current.len()].join("/"),
                path = if it.def_id.is_local() {
                    cx.current.join("/")
                } else {
                    let path = &cache.external_paths[&it.def_id];
                    path[..path.len() - 1].join("/")
                },
                ty = shortty(it).to_static_str(),
                name = *it.name.as_ref().unwrap()));
    Ok(())
}

fn assoc_const(w: &mut fmt::Formatter, it: &clean::Item,
               ty: &clean::Type, default: Option<&String>)
               -> fmt::Result {
    try!(write!(w, "const {}", it.name.as_ref().unwrap()));
    try!(write!(w, ": {}", ty));
    if let Some(default) = default {
        try!(write!(w, " = {}", default));
    }
    Ok(())
}

fn assoc_type(w: &mut fmt::Formatter, it: &clean::Item,
              bounds: &Vec<clean::TyParamBound>,
              default: &Option<clean::Type>)
              -> fmt::Result {
    try!(write!(w, "type {}", it.name.as_ref().unwrap()));
    if !bounds.is_empty() {
        try!(write!(w, ": {}", TyParamBounds(bounds)))
    }
    if let Some(ref default) = *default {
        try!(write!(w, " = {}", default));
    }
    Ok(())
}

fn render_assoc_item(w: &mut fmt::Formatter, meth: &clean::Item,
                     link: AssocItemLink) -> fmt::Result {
    fn method(w: &mut fmt::Formatter,
              it: &clean::Item,
              unsafety: hir::Unsafety,
              constness: hir::Constness,
              abi: abi::Abi,
              g: &clean::Generics,
              selfty: &clean::SelfTy,
              d: &clean::FnDecl,
              link: AssocItemLink)
              -> fmt::Result {
        use syntax::abi::Abi;

        let name = it.name.as_ref().unwrap();
        let anchor = format!("#{}.{}", shortty(it), name);
        let href = match link {
            AssocItemLink::Anchor => anchor,
            AssocItemLink::GotoSource(did) => {
                href(did).map(|p| format!("{}{}", p.0, anchor)).unwrap_or(anchor)
            }
        };
        write!(w, "{}{}{}fn <a href='{href}' class='fnname'>{name}</a>\
                   {generics}{decl}{where_clause}",
               ConstnessSpace(constness),
               UnsafetySpace(unsafety),
               match abi {
                   Abi::Rust => String::new(),
                   a => format!("extern {} ", a.to_string())
               },
               href = href,
               name = name,
               generics = *g,
               decl = Method(selfty, d),
               where_clause = WhereClause(g))
    }
    match meth.inner {
        clean::TyMethodItem(ref m) => {
            method(w, meth, m.unsafety, hir::Constness::NotConst,
                   m.abi, &m.generics, &m.self_, &m.decl, link)
        }
        clean::MethodItem(ref m) => {
            method(w, meth, m.unsafety, m.constness,
                   m.abi, &m.generics, &m.self_, &m.decl,
                   link)
        }
        clean::AssociatedConstItem(ref ty, ref default) => {
            assoc_const(w, meth, ty, default.as_ref())
        }
        clean::AssociatedTypeItem(ref bounds, ref default) => {
            assoc_type(w, meth, bounds, default)
        }
        _ => panic!("render_assoc_item called on non-associated-item")
    }
}

fn item_struct(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
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

    try!(document(w, cx, it));
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
                let name = field.name.as_ref().unwrap();
                try!(with_unique_id(format!("structfield.{}", name), |id|
                    write!(w, "<tr class='stab {}'><td id='{}'><code>{}</code></td><td>",
                              field.stability_class(),
                              id,
                              name)));
                try!(document(w, cx, field));
                try!(write!(w, "</td></tr>"));
            }
            try!(write!(w, "</table>"));
        }
    }
    render_assoc_items(w, cx, it.def_id, AssocItemRender::All)
}

fn item_enum(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
             e: &clean::Enum) -> fmt::Result {
    try!(write!(w, "<pre class='rust enum'>"));
    try!(render_attributes(w, it));
    try!(write!(w, "{}enum {}{}{}",
                  VisSpace(it.visibility),
                  it.name.as_ref().unwrap(),
                  e.generics,
                  WhereClause(&e.generics)));
    if e.variants.is_empty() && !e.variants_stripped {
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

    try!(document(w, cx, it));
    if !e.variants.is_empty() {
        try!(write!(w, "<h2 class='variants'>Variants</h2>\n<table>"));
        for variant in &e.variants {
            let name = variant.name.as_ref().unwrap();
            try!(with_unique_id(format!("variant.{}", name), |id|
                    write!(w, "<tr><td id='{}'><code>{}</code></td><td>", id, name)));
            try!(document(w, cx, variant));
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
                                let v = variant.name.as_ref().unwrap();
                                let f = field.name.as_ref().unwrap();
                                try!(with_unique_id(format!("variant.{}.field.{}", v, f), |id|
                                    write!(w, "<tr><td id='{}'><code>{}</code></td><td>", id, f)));
                                try!(document(w, cx, field));
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
    try!(render_assoc_items(w, cx, it.def_id, AssocItemRender::All));
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

#[derive(Copy, Clone)]
enum AssocItemLink {
    Anchor,
    GotoSource(DefId),
}

enum AssocItemRender<'a> {
    All,
    DerefFor { trait_: &'a clean::Type, type_: &'a clean::Type },
}

fn render_assoc_items(w: &mut fmt::Formatter,
                      cx: &Context,
                      it: DefId,
                      what: AssocItemRender) -> fmt::Result {
    let c = cache();
    let v = match c.impls.get(&it) {
        Some(v) => v,
        None => return Ok(()),
    };
    let (non_trait, traits): (Vec<_>, _) = v.iter().partition(|i| {
        i.impl_.trait_.is_none()
    });
    if !non_trait.is_empty() {
        let render_header = match what {
            AssocItemRender::All => {
                try!(write!(w, "<h2 id='methods'>Methods</h2>"));
                true
            }
            AssocItemRender::DerefFor { trait_, type_ } => {
                try!(write!(w, "<h2 id='deref-methods'>Methods from \
                                    {}&lt;Target={}&gt;</h2>", trait_, type_));
                false
            }
        };
        for i in &non_trait {
            try!(render_impl(w, cx, i, AssocItemLink::Anchor, render_header));
        }
    }
    if let AssocItemRender::DerefFor { .. } = what {
        return Ok(())
    }
    if !traits.is_empty() {
        let deref_impl = traits.iter().find(|t| {
            match *t.impl_.trait_.as_ref().unwrap() {
                clean::ResolvedPath { did, .. } => {
                    Some(did) == c.deref_trait_did
                }
                _ => false
            }
        });
        if let Some(impl_) = deref_impl {
            try!(render_deref_methods(w, cx, impl_));
        }
        try!(write!(w, "<h2 id='implementations'>Trait \
                          Implementations</h2>"));
        let (derived, manual): (Vec<_>, Vec<&Impl>) = traits.iter().partition(|i| {
            i.impl_.derived
        });
        for i in &manual {
            let did = i.trait_did().unwrap();
            try!(render_impl(w, cx, i, AssocItemLink::GotoSource(did), true));
        }
        if !derived.is_empty() {
            try!(write!(w, "<h3 id='derived_implementations'>\
                Derived Implementations \
            </h3>"));
            for i in &derived {
                let did = i.trait_did().unwrap();
                try!(render_impl(w, cx, i, AssocItemLink::GotoSource(did), true));
            }
        }
    }
    Ok(())
}

fn render_deref_methods(w: &mut fmt::Formatter, cx: &Context, impl_: &Impl) -> fmt::Result {
    let deref_type = impl_.impl_.trait_.as_ref().unwrap();
    let target = impl_.impl_.items.iter().filter_map(|item| {
        match item.inner {
            clean::TypedefItem(ref t, true) => Some(&t.type_),
            _ => None,
        }
    }).next().expect("Expected associated type binding");
    let what = AssocItemRender::DerefFor { trait_: deref_type, type_: target };
    match *target {
        clean::ResolvedPath { did, .. } => render_assoc_items(w, cx, did, what),
        _ => {
            if let Some(prim) = target.primitive_type() {
                if let Some(c) = cache().primitive_locations.get(&prim) {
                    let did = DefId { krate: *c, index: prim.to_def_index() };
                    try!(render_assoc_items(w, cx, did, what));
                }
            }
            Ok(())
        }
    }
}

// Render_header is false when we are rendering a `Deref` impl and true
// otherwise. If render_header is false, we will avoid rendering static
// methods, since they are not accessible for the type implementing `Deref`
fn render_impl(w: &mut fmt::Formatter, cx: &Context, i: &Impl, link: AssocItemLink,
               render_header: bool) -> fmt::Result {
    if render_header {
        try!(write!(w, "<h3 class='impl'><code>{}</code></h3>", i.impl_));
        if let Some(ref dox) = i.dox {
            try!(write!(w, "<div class='docblock'>{}</div>", Markdown(dox)));
        }
    }

    fn doctraititem(w: &mut fmt::Formatter, cx: &Context, item: &clean::Item,
                    link: AssocItemLink, render_static: bool) -> fmt::Result {
        let name = item.name.as_ref().unwrap();
        match item.inner {
            clean::MethodItem(..) | clean::TyMethodItem(..) => {
                // Only render when the method is not static or we allow static methods
                if !is_static_method(item) || render_static {
                    try!(with_unique_id(format!("method.{}", name), |id|
                        write!(w, "<h4 id='{}' class='{}'><code>", id, shortty(item))));
                try!(render_assoc_item(w, item, link));
                    try!(write!(w, "</code></h4>\n"));
                }
            }
            clean::TypedefItem(ref tydef, _) => {
                try!(with_unique_id(format!("assoc_type.{}", name), |id|
                    write!(w, "<h4 id='{}' class='{}'><code>", id, shortty(item))));
                try!(write!(w, "type {} = {}", name, tydef.type_));
                try!(write!(w, "</code></h4>\n"));
            }
            clean::AssociatedConstItem(ref ty, ref default) => {
                try!(with_unique_id(format!("assoc_const.{}", name), |id|
                    write!(w, "<h4 id='{}' class='{}'><code>", id, shortty(item))));
                try!(assoc_const(w, item, ty, default.as_ref()));
                try!(write!(w, "</code></h4>\n"));
            }
            clean::ConstantItem(ref c) => {
                try!(with_unique_id(format!("assoc_const.{}", name), |id|
                    write!(w, "<h4 id='{}' class='{}'><code>", id, shortty(item))));
                try!(assoc_const(w, item, &c.type_, Some(&c.expr)));
                try!(write!(w, "</code></h4>\n"));
            }
            clean::AssociatedTypeItem(ref bounds, ref default) => {
                try!(with_unique_id(format!("assoc_type.{}", name), |id|
                    write!(w, "<h4 id='{}' class='{}'><code>", id, shortty(item))));
                try!(assoc_type(w, item, bounds, default));
                try!(write!(w, "</code></h4>\n"));
            }
            _ => panic!("can't make docs for trait item with name {:?}", item.name)
        }

        return if let AssocItemLink::Anchor = link {
            if is_static_method(item) && !render_static {
                Ok(())
            } else {
                document(w, cx, item)
            }
        } else {
            Ok(())
        };

        fn is_static_method(item: &clean::Item) -> bool {
            match item.inner {
                clean::MethodItem(ref method) => method.self_ == SelfTy::SelfStatic,
                clean::TyMethodItem(ref method) => method.self_ == SelfTy::SelfStatic,
                _ => false
            }
        }
    }

    try!(write!(w, "<div class='impl-items'>"));
    for trait_item in &i.impl_.items {
        try!(doctraititem(w, cx, trait_item, link, render_header));
    }

    fn render_default_items(w: &mut fmt::Formatter,
                            cx: &Context,
                            did: DefId,
                            t: &clean::Trait,
                              i: &clean::Impl,
                              render_static: bool) -> fmt::Result {
        for trait_item in &t.items {
            let n = trait_item.name.clone();
            match i.items.iter().find(|m| { m.name == n }) {
                Some(..) => continue,
                None => {}
            }

            try!(doctraititem(w, cx, trait_item, AssocItemLink::GotoSource(did), render_static));
        }
        Ok(())
    }

    // If we've implemented a trait, then also emit documentation for all
    // default methods which weren't overridden in the implementation block.
    // FIXME: this also needs to be done for associated types, whenever defaults
    // for them work.
    if let Some(clean::ResolvedPath { did, .. }) = i.impl_.trait_ {
        if let Some(t) = cache().traits.get(&did) {
            try!(render_default_items(w, cx, did, t, &i.impl_, render_header));

        }
    }
    try!(write!(w, "</div>"));
    Ok(())
}

fn item_typedef(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
                t: &clean::Typedef) -> fmt::Result {
    try!(write!(w, "<pre class='rust typedef'>type {}{}{where_clause} = {type_};</pre>",
                  it.name.as_ref().unwrap(),
                  t.generics,
                  where_clause = WhereClause(&t.generics),
                  type_ = t.type_));

    document(w, cx, it)
}

impl<'a> fmt::Display for Sidebar<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let cx = self.cx;
        let it = self.item;
        let parentlen = cx.current.len() - if it.is_mod() {1} else {0};

        // the sidebar is designed to display sibling functions, modules and
        // other miscellaneous informations. since there are lots of sibling
        // items (and that causes quadratic growth in large modules),
        // we refactor common parts into a shared JavaScript file per module.
        // still, we don't move everything into JS because we want to preserve
        // as much HTML as possible in order to allow non-JS-enabled browsers
        // to navigate the documentation (though slightly inefficiently).

        try!(write!(fmt, "<p class='location'>"));
        for (i, name) in cx.current.iter().take(parentlen).enumerate() {
            if i > 0 {
                try!(write!(fmt, "::<wbr>"));
            }
            try!(write!(fmt, "<a href='{}index.html'>{}</a>",
                          &cx.root_path[..(cx.current.len() - i - 1) * 3],
                          *name));
        }
        try!(write!(fmt, "</p>"));

        // sidebar refers to the enclosing module, not this module
        let relpath = if shortty(it) == ItemType::Module { "../" } else { "" };
        try!(write!(fmt,
                    "<script>window.sidebarCurrent = {{\
                        name: '{name}', \
                        ty: '{ty}', \
                        relpath: '{path}'\
                     }};</script>",
                    name = it.name.as_ref().map(|x| &x[..]).unwrap_or(""),
                    ty = shortty(it).to_static_str(),
                    path = relpath));
        if parentlen == 0 {
            // there is no sidebar-items.js beyond the crate root path
            // FIXME maybe dynamic crate loading can be merged here
        } else {
            try!(write!(fmt, "<script defer src=\"{path}sidebar-items.js\"></script>",
                        path = relpath));
        }

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

fn item_macro(w: &mut fmt::Formatter, cx: &Context, it: &clean::Item,
              t: &clean::Macro) -> fmt::Result {
    try!(w.write_str(&highlight::highlight(&t.source,
                                          Some("macro"),
                                          None)));
    document(w, cx, it)
}

fn item_primitive(w: &mut fmt::Formatter, cx: &Context,
                  it: &clean::Item,
                  _p: &clean::PrimitiveType) -> fmt::Result {
    try!(document(w, cx, it));
    render_assoc_items(w, cx, it.def_id, AssocItemRender::All)
}

fn get_basic_keywords() -> &'static str {
    "rust, rustlang, rust-lang"
}

fn make_item_keywords(it: &clean::Item) -> String {
    format!("{}, {}", get_basic_keywords(), it.name.as_ref().unwrap())
}

fn get_index_search_type(item: &clean::Item,
                         parent: Option<String>) -> Option<IndexItemFunctionType> {
    let decl = match item.inner {
        clean::FunctionItem(ref f) => &f.decl,
        clean::MethodItem(ref m) => &m.decl,
        clean::TyMethodItem(ref m) => &m.decl,
        _ => return None
    };

    let mut inputs = Vec::new();

    // Consider `self` an argument as well.
    if let Some(name) = parent {
        inputs.push(Type { name: Some(name.to_ascii_lowercase()) });
    }

    inputs.extend(&mut decl.inputs.values.iter().map(|arg| {
        get_index_type(&arg.type_)
    }));

    let output = match decl.output {
        clean::FunctionRetTy::Return(ref return_type) => Some(get_index_type(return_type)),
        _ => None
    };

    Some(IndexItemFunctionType { inputs: inputs, output: output })
}

fn get_index_type(clean_type: &clean::Type) -> Type {
    Type { name: get_index_type_name(clean_type).map(|s| s.to_ascii_lowercase()) }
}

fn get_index_type_name(clean_type: &clean::Type) -> Option<String> {
    match *clean_type {
        clean::ResolvedPath { ref path, .. } => {
            let segments = &path.segments;
            Some(segments[segments.len() - 1].name.clone())
        },
        clean::Generic(ref s) => Some(s.clone()),
        clean::Primitive(ref p) => Some(format!("{:?}", p)),
        clean::BorrowedRef { ref type_, .. } => get_index_type_name(type_),
        // FIXME: add all from clean::Type.
        _ => None
    }
}

pub fn cache() -> Arc<Cache> {
    CACHE_KEY.with(|c| c.borrow().clone())
}
