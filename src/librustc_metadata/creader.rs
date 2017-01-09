// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Validates all used crates and extern libraries and loads their metadata

use cstore::{self, CStore, CrateSource, MetadataBlob};
use locator::{self, CratePaths};
use schema::CrateRoot;

use rustc::hir::def_id::{CrateNum, DefIndex};
use rustc::hir::svh::Svh;
use rustc::middle::cstore::DepKind;
use rustc::session::{config, Session};
use rustc_back::PanicStrategy;
use rustc::session::search_paths::PathKind;
use rustc::middle;
use rustc::middle::cstore::{CrateStore, validate_crate_name, ExternCrate};
use rustc::util::common::record_time;
use rustc::util::nodemap::FxHashSet;
use rustc::middle::cstore::NativeLibrary;
use rustc::hir::map::Definitions;

use std::cell::{RefCell, Cell};
use std::ops::Deref;
use std::path::PathBuf;
use std::rc::Rc;
use std::{cmp, fs};

use syntax::ast;
use syntax::abi::Abi;
use syntax::attr;
use syntax::ext::base::SyntaxExtension;
use syntax::feature_gate::{self, GateIssue};
use syntax::symbol::Symbol;
use syntax_pos::{Span, DUMMY_SP};
use log;

pub struct Library {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub rmeta: Option<(PathBuf, PathKind)>,
    pub metadata: MetadataBlob,
}

pub struct CrateLoader<'a> {
    pub sess: &'a Session,
    cstore: &'a CStore,
    next_crate_num: CrateNum,
    local_crate_name: Symbol,
}

fn dump_crates(cstore: &CStore) {
    info!("resolved crates:");
    cstore.iter_crate_data(|_, data| {
        info!("  name: {}", data.name());
        info!("  cnum: {}", data.cnum);
        info!("  hash: {}", data.hash());
        info!("  reqd: {:?}", data.dep_kind.get());
        let CrateSource { dylib, rlib, rmeta } = data.source.clone();
        dylib.map(|dl| info!("  dylib: {}", dl.0.display()));
        rlib.map(|rl|  info!("   rlib: {}", rl.0.display()));
        rmeta.map(|rl| info!("   rmeta: {}", rl.0.display()));
    });
}

#[derive(Debug)]
struct ExternCrateInfo {
    ident: Symbol,
    name: Symbol,
    id: ast::NodeId,
    dep_kind: DepKind,
}

fn register_native_lib(sess: &Session,
                       cstore: &CStore,
                       span: Option<Span>,
                       lib: NativeLibrary) {
    if lib.name.as_str().is_empty() {
        match span {
            Some(span) => {
                struct_span_err!(sess, span, E0454,
                                 "#[link(name = \"\")] given with empty name")
                    .span_label(span, &format!("empty name given"))
                    .emit();
            }
            None => {
                sess.err("empty library name given via `-l`");
            }
        }
        return
    }
    let is_osx = sess.target.target.options.is_like_osx;
    if lib.kind == cstore::NativeFramework && !is_osx {
        let msg = "native frameworks are only available on OSX targets";
        match span {
            Some(span) => span_err!(sess, span, E0455, "{}", msg),
            None => sess.err(msg),
        }
    }
    if lib.cfg.is_some() && !sess.features.borrow().link_cfg {
        feature_gate::emit_feature_err(&sess.parse_sess,
                                       "link_cfg",
                                       span.unwrap(),
                                       GateIssue::Language,
                                       "is feature gated");
    }
    cstore.add_used_library(lib);
}

fn relevant_lib(sess: &Session, lib: &NativeLibrary) -> bool {
    match lib.cfg {
        Some(ref cfg) => attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

// Extra info about a crate loaded for plugins or exported macros.
struct ExtensionCrate {
    metadata: PMDSource,
    dylib: Option<PathBuf>,
    target_only: bool,
}

enum PMDSource {
    Registered(Rc<cstore::CrateMetadata>),
    Owned(Library),
}

impl Deref for PMDSource {
    type Target = MetadataBlob;

    fn deref(&self) -> &MetadataBlob {
        match *self {
            PMDSource::Registered(ref cmd) => &cmd.blob,
            PMDSource::Owned(ref lib) => &lib.metadata
        }
    }
}

enum LoadResult {
    Previous(CrateNum),
    Loaded(Library),
}

impl<'a> CrateLoader<'a> {
    pub fn new(sess: &'a Session, cstore: &'a CStore, local_crate_name: &str) -> Self {
        CrateLoader {
            sess: sess,
            cstore: cstore,
            next_crate_num: cstore.next_crate_num(),
            local_crate_name: Symbol::intern(local_crate_name),
        }
    }

    fn extract_crate_info(&self, i: &ast::Item) -> Option<ExternCrateInfo> {
        match i.node {
            ast::ItemKind::ExternCrate(ref path_opt) => {
                debug!("resolving extern crate stmt. ident: {} path_opt: {:?}",
                       i.ident, path_opt);
                let name = match *path_opt {
                    Some(name) => {
                        validate_crate_name(Some(self.sess), &name.as_str(),
                                            Some(i.span));
                        name
                    }
                    None => i.ident.name,
                };
                Some(ExternCrateInfo {
                    ident: i.ident.name,
                    name: name,
                    id: i.id,
                    dep_kind: if attr::contains_name(&i.attrs, "no_link") {
                        DepKind::UnexportedMacrosOnly
                    } else {
                        DepKind::Explicit
                    },
                })
            }
            _ => None
        }
    }

    fn existing_match(&self, name: Symbol, hash: Option<&Svh>, kind: PathKind)
                      -> Option<CrateNum> {
        let mut ret = None;
        self.cstore.iter_crate_data(|cnum, data| {
            if data.name != name { return }

            match hash {
                Some(hash) if *hash == data.hash() => { ret = Some(cnum); return }
                Some(..) => return,
                None => {}
            }

            // When the hash is None we're dealing with a top-level dependency
            // in which case we may have a specification on the command line for
            // this library. Even though an upstream library may have loaded
            // something of the same name, we have to make sure it was loaded
            // from the exact same location as well.
            //
            // We're also sure to compare *paths*, not actual byte slices. The
            // `source` stores paths which are normalized which may be different
            // from the strings on the command line.
            let source = self.cstore.used_crate_source(cnum);
            if let Some(locs) = self.sess.opts.externs.get(&*name.as_str()) {
                let found = locs.iter().any(|l| {
                    let l = fs::canonicalize(l).ok();
                    source.dylib.as_ref().map(|p| &p.0) == l.as_ref() ||
                    source.rlib.as_ref().map(|p| &p.0) == l.as_ref()
                });
                if found {
                    ret = Some(cnum);
                }
                return
            }

            // Alright, so we've gotten this far which means that `data` has the
            // right name, we don't have a hash, and we don't have a --extern
            // pointing for ourselves. We're still not quite yet done because we
            // have to make sure that this crate was found in the crate lookup
            // path (this is a top-level dependency) as we don't want to
            // implicitly load anything inside the dependency lookup path.
            let prev_kind = source.dylib.as_ref().or(source.rlib.as_ref())
                                  .unwrap().1;
            if ret.is_none() && (prev_kind == kind || prev_kind == PathKind::All) {
                ret = Some(cnum);
            }
        });
        return ret;
    }

    fn verify_no_symbol_conflicts(&self,
                                  span: Span,
                                  root: &CrateRoot) {
        // Check for (potential) conflicts with the local crate
        if self.local_crate_name == root.name &&
           self.sess.local_crate_disambiguator() == root.disambiguator {
            span_fatal!(self.sess, span, E0519,
                        "the current crate is indistinguishable from one of its \
                         dependencies: it has the same crate-name `{}` and was \
                         compiled with the same `-C metadata` arguments. This \
                         will result in symbol conflicts between the two.",
                        root.name)
        }

        // Check for conflicts with any crate loaded so far
        self.cstore.iter_crate_data(|_, other| {
            if other.name() == root.name && // same crate-name
               other.disambiguator() == root.disambiguator &&  // same crate-disambiguator
               other.hash() != root.hash { // but different SVH
                span_fatal!(self.sess, span, E0523,
                        "found two different crates with name `{}` that are \
                         not distinguished by differing `-C metadata`. This \
                         will result in symbol conflicts between the two.",
                        root.name)
            }
        });
    }

    fn register_crate(&mut self,
                      root: &Option<CratePaths>,
                      ident: Symbol,
                      name: Symbol,
                      span: Span,
                      lib: Library,
                      dep_kind: DepKind)
                      -> (CrateNum, Rc<cstore::CrateMetadata>) {
        info!("register crate `extern crate {} as {}`", name, ident);
        let crate_root = lib.metadata.get_root();
        self.verify_no_symbol_conflicts(span, &crate_root);

        // Claim this crate number and cache it
        let cnum = self.next_crate_num;
        self.next_crate_num = CrateNum::from_u32(cnum.as_u32() + 1);

        // Stash paths for top-most crate locally if necessary.
        let crate_paths = if root.is_none() {
            Some(CratePaths {
                ident: ident.to_string(),
                dylib: lib.dylib.clone().map(|p| p.0),
                rlib:  lib.rlib.clone().map(|p| p.0),
                rmeta: lib.rmeta.clone().map(|p| p.0),
            })
        } else {
            None
        };
        // Maintain a reference to the top most crate.
        let root = if root.is_some() { root } else { &crate_paths };

        let Library { dylib, rlib, rmeta, metadata } = lib;

        let cnum_map = self.resolve_crate_deps(root, &crate_root, &metadata, cnum, span, dep_kind);

        let def_path_table = record_time(&self.sess.perf_stats.decode_def_path_tables_time, || {
            crate_root.def_path_table.decode(&metadata)
        });

        let exported_symbols = crate_root.exported_symbols.decode(&metadata).collect();

        let mut cmeta = cstore::CrateMetadata {
            name: name,
            extern_crate: Cell::new(None),
            def_path_table: def_path_table,
            exported_symbols: exported_symbols,
            proc_macros: crate_root.macro_derive_registrar.map(|_| {
                self.load_derive_macros(&crate_root, dylib.clone().map(|p| p.0), span)
            }),
            root: crate_root,
            blob: metadata,
            cnum_map: RefCell::new(cnum_map),
            cnum: cnum,
            codemap_import_info: RefCell::new(vec![]),
            dep_kind: Cell::new(dep_kind),
            source: cstore::CrateSource {
                dylib: dylib,
                rlib: rlib,
                rmeta: rmeta,
            },
            dllimport_foreign_items: FxHashSet(),
        };

        let dllimports: Vec<_> = cmeta.get_native_libraries().iter()
                            .filter(|lib| relevant_lib(self.sess, lib) &&
                                          lib.kind == cstore::NativeLibraryKind::NativeUnknown)
                            .flat_map(|lib| &lib.foreign_items)
                            .map(|id| *id)
                            .collect();
        cmeta.dllimport_foreign_items.extend(dllimports);

        let cmeta = Rc::new(cmeta);
        self.cstore.set_crate_data(cnum, cmeta.clone());
        (cnum, cmeta)
    }

    fn resolve_crate(&mut self,
                     root: &Option<CratePaths>,
                     ident: Symbol,
                     name: Symbol,
                     hash: Option<&Svh>,
                     span: Span,
                     path_kind: PathKind,
                     mut dep_kind: DepKind)
                     -> (CrateNum, Rc<cstore::CrateMetadata>) {
        info!("resolving crate `extern crate {} as {}`", name, ident);
        let result = if let Some(cnum) = self.existing_match(name, hash, path_kind) {
            LoadResult::Previous(cnum)
        } else {
            info!("falling back to a load");
            let mut locate_ctxt = locator::Context {
                sess: self.sess,
                span: span,
                ident: ident,
                crate_name: name,
                hash: hash.map(|a| &*a),
                filesearch: self.sess.target_filesearch(path_kind),
                target: &self.sess.target.target,
                triple: &self.sess.opts.target_triple,
                root: root,
                rejected_via_hash: vec![],
                rejected_via_triple: vec![],
                rejected_via_kind: vec![],
                rejected_via_version: vec![],
                rejected_via_filename: vec![],
                should_match_name: true,
                is_proc_macro: Some(false),
            };

            self.load(&mut locate_ctxt).or_else(|| {
                dep_kind = DepKind::UnexportedMacrosOnly;

                let mut proc_macro_locator = locator::Context {
                    target: &self.sess.host,
                    triple: config::host_triple(),
                    filesearch: self.sess.host_filesearch(path_kind),
                    rejected_via_hash: vec![],
                    rejected_via_triple: vec![],
                    rejected_via_kind: vec![],
                    rejected_via_version: vec![],
                    rejected_via_filename: vec![],
                    is_proc_macro: Some(true),
                    ..locate_ctxt
                };

                self.load(&mut proc_macro_locator)
            }).unwrap_or_else(|| locate_ctxt.report_errs())
        };

        match result {
            LoadResult::Previous(cnum) => {
                let data = self.cstore.get_crate_data(cnum);
                if data.root.macro_derive_registrar.is_some() {
                    dep_kind = DepKind::UnexportedMacrosOnly;
                }
                data.dep_kind.set(cmp::max(data.dep_kind.get(), dep_kind));
                (cnum, data)
            }
            LoadResult::Loaded(library) => {
                self.register_crate(root, ident, name, span, library, dep_kind)
            }
        }
    }

    fn load(&mut self, locate_ctxt: &mut locator::Context) -> Option<LoadResult> {
        let library = match locate_ctxt.maybe_load_library_crate() {
            Some(lib) => lib,
            None => return None,
        };

        // In the case that we're loading a crate, but not matching
        // against a hash, we could load a crate which has the same hash
        // as an already loaded crate. If this is the case prevent
        // duplicates by just using the first crate.
        //
        // Note that we only do this for target triple crates, though, as we
        // don't want to match a host crate against an equivalent target one
        // already loaded.
        let root = library.metadata.get_root();
        if locate_ctxt.triple == self.sess.opts.target_triple {
            let mut result = LoadResult::Loaded(library);
            self.cstore.iter_crate_data(|cnum, data| {
                if data.name() == root.name && root.hash == data.hash() {
                    assert!(locate_ctxt.hash.is_none());
                    info!("load success, going to previous cnum: {}", cnum);
                    result = LoadResult::Previous(cnum);
                }
            });
            Some(result)
        } else {
            Some(LoadResult::Loaded(library))
        }
    }

    fn update_extern_crate(&mut self,
                           cnum: CrateNum,
                           mut extern_crate: ExternCrate,
                           visited: &mut FxHashSet<(CrateNum, bool)>)
    {
        if !visited.insert((cnum, extern_crate.direct)) { return }

        let cmeta = self.cstore.get_crate_data(cnum);
        let old_extern_crate = cmeta.extern_crate.get();

        // Prefer:
        // - something over nothing (tuple.0);
        // - direct extern crate to indirect (tuple.1);
        // - shorter paths to longer (tuple.2).
        let new_rank = (true, extern_crate.direct, !extern_crate.path_len);
        let old_rank = match old_extern_crate {
            None => (false, false, !0),
            Some(ref c) => (true, c.direct, !c.path_len),
        };

        if old_rank >= new_rank {
            return; // no change needed
        }

        cmeta.extern_crate.set(Some(extern_crate));
        // Propagate the extern crate info to dependencies.
        extern_crate.direct = false;
        for &dep_cnum in cmeta.cnum_map.borrow().iter() {
            self.update_extern_crate(dep_cnum, extern_crate, visited);
        }
    }

    // Go through the crate metadata and load any crates that it references
    fn resolve_crate_deps(&mut self,
                          root: &Option<CratePaths>,
                          crate_root: &CrateRoot,
                          metadata: &MetadataBlob,
                          krate: CrateNum,
                          span: Span,
                          dep_kind: DepKind)
                          -> cstore::CrateNumMap {
        debug!("resolving deps of external crate");
        if crate_root.macro_derive_registrar.is_some() {
            return cstore::CrateNumMap::new();
        }

        // The map from crate numbers in the crate we're resolving to local crate numbers.
        // We map 0 and all other holes in the map to our parent crate. The "additional"
        // self-dependencies should be harmless.
        ::std::iter::once(krate).chain(crate_root.crate_deps.decode(metadata).map(|dep| {
            debug!("resolving dep crate {} hash: `{}`", dep.name, dep.hash);
            if dep.kind == DepKind::UnexportedMacrosOnly {
                return krate;
            }
            let dep_kind = match dep_kind {
                DepKind::MacrosOnly => DepKind::MacrosOnly,
                _ => dep.kind,
            };
            let (local_cnum, ..) = self.resolve_crate(
                root, dep.name, dep.name, Some(&dep.hash), span, PathKind::Dependency, dep_kind,
            );
            local_cnum
        })).collect()
    }

    fn read_extension_crate(&mut self, span: Span, info: &ExternCrateInfo) -> ExtensionCrate {
        info!("read extension crate {} `extern crate {} as {}` dep_kind={:?}",
              info.id, info.name, info.ident, info.dep_kind);
        let target_triple = &self.sess.opts.target_triple[..];
        let is_cross = target_triple != config::host_triple();
        let mut target_only = false;
        let mut locate_ctxt = locator::Context {
            sess: self.sess,
            span: span,
            ident: info.ident,
            crate_name: info.name,
            hash: None,
            filesearch: self.sess.host_filesearch(PathKind::Crate),
            target: &self.sess.host,
            triple: config::host_triple(),
            root: &None,
            rejected_via_hash: vec![],
            rejected_via_triple: vec![],
            rejected_via_kind: vec![],
            rejected_via_version: vec![],
            rejected_via_filename: vec![],
            should_match_name: true,
            is_proc_macro: None,
        };
        let library = self.load(&mut locate_ctxt).or_else(|| {
            if !is_cross {
                return None
            }
            // Try loading from target crates. This will abort later if we
            // try to load a plugin registrar function,
            target_only = true;

            locate_ctxt.target = &self.sess.target.target;
            locate_ctxt.triple = target_triple;
            locate_ctxt.filesearch = self.sess.target_filesearch(PathKind::Crate);

            self.load(&mut locate_ctxt)
        });
        let library = match library {
            Some(l) => l,
            None => locate_ctxt.report_errs(),
        };

        let (dylib, metadata) = match library {
            LoadResult::Previous(cnum) => {
                let data = self.cstore.get_crate_data(cnum);
                (data.source.dylib.clone(), PMDSource::Registered(data))
            }
            LoadResult::Loaded(library) => {
                let dylib = library.dylib.clone();
                let metadata = PMDSource::Owned(library);
                (dylib, metadata)
            }
        };

        ExtensionCrate {
            metadata: metadata,
            dylib: dylib.map(|p| p.0),
            target_only: target_only,
        }
    }

    /// Load custom derive macros.
    ///
    /// Note that this is intentionally similar to how we load plugins today,
    /// but also intentionally separate. Plugins are likely always going to be
    /// implemented as dynamic libraries, but we have a possible future where
    /// custom derive (and other macro-1.1 style features) are implemented via
    /// executables and custom IPC.
    fn load_derive_macros(&mut self, root: &CrateRoot, dylib: Option<PathBuf>, span: Span)
                          -> Vec<(ast::Name, Rc<SyntaxExtension>)> {
        use std::{env, mem};
        use proc_macro::TokenStream;
        use proc_macro::__internal::Registry;
        use rustc_back::dynamic_lib::DynamicLibrary;
        use syntax_ext::deriving::custom::CustomDerive;
        use syntax_ext::proc_macro_impl::AttrProcMacro;

        let path = match dylib {
            Some(dylib) => dylib,
            None => span_bug!(span, "proc-macro crate not dylib"),
        };
        // Make sure the path contains a / or the linker will search for it.
        let path = env::current_dir().unwrap().join(path);
        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            Err(err) => self.sess.span_fatal(span, &err),
        };

        let sym = self.sess.generate_derive_registrar_symbol(&root.hash,
                                                             root.macro_derive_registrar.unwrap());
        let registrar = unsafe {
            let sym = match lib.symbol(&sym) {
                Ok(f) => f,
                Err(err) => self.sess.span_fatal(span, &err),
            };
            mem::transmute::<*mut u8, fn(&mut Registry)>(sym)
        };

        struct MyRegistrar(Vec<(ast::Name, Rc<SyntaxExtension>)>);

        impl Registry for MyRegistrar {
            fn register_custom_derive(&mut self,
                                      trait_name: &str,
                                      expand: fn(TokenStream) -> TokenStream,
                                      attributes: &[&'static str]) {
                let attrs = attributes.iter().cloned().map(Symbol::intern).collect();
                let derive = SyntaxExtension::CustomDerive(
                    Box::new(CustomDerive::new(expand, attrs))
                );
                self.0.push((Symbol::intern(trait_name), Rc::new(derive)));
            }

            fn register_attr_proc_macro(&mut self,
                                        name: &str,
                                        expand: fn(TokenStream, TokenStream) -> TokenStream) {
                let expand = SyntaxExtension::AttrProcMacro(
                    Box::new(AttrProcMacro { inner: expand })
                );
                self.0.push((Symbol::intern(name), Rc::new(expand)));
            }
        }

        let mut my_registrar = MyRegistrar(Vec::new());
        registrar(&mut my_registrar);

        // Intentionally leak the dynamic library. We can't ever unload it
        // since the library can make things that will live arbitrarily long.
        mem::forget(lib);
        my_registrar.0
    }

    /// Look for a plugin registrar. Returns library path, crate
    /// SVH and DefIndex of the registrar function.
    pub fn find_plugin_registrar(&mut self, span: Span, name: &str)
                                 -> Option<(PathBuf, Svh, DefIndex)> {
        let ekrate = self.read_extension_crate(span, &ExternCrateInfo {
             name: Symbol::intern(name),
             ident: Symbol::intern(name),
             id: ast::DUMMY_NODE_ID,
             dep_kind: DepKind::UnexportedMacrosOnly,
        });

        if ekrate.target_only {
            // Need to abort before syntax expansion.
            let message = format!("plugin `{}` is not available for triple `{}` \
                                   (only found {})",
                                  name,
                                  config::host_triple(),
                                  self.sess.opts.target_triple);
            span_fatal!(self.sess, span, E0456, "{}", &message[..]);
        }

        let root = ekrate.metadata.get_root();
        match (ekrate.dylib.as_ref(), root.plugin_registrar_fn) {
            (Some(dylib), Some(reg)) => {
                Some((dylib.to_path_buf(), root.hash, reg))
            }
            (None, Some(_)) => {
                span_err!(self.sess, span, E0457,
                          "plugin `{}` only found in rlib format, but must be available \
                           in dylib format",
                          name);
                // No need to abort because the loading code will just ignore this
                // empty dylib.
                None
            }
            _ => None,
        }
    }

    fn get_foreign_items_of_kind(&self, kind: cstore::NativeLibraryKind) -> Vec<DefIndex> {
        let mut items = vec![];
        let libs = self.cstore.get_used_libraries();
        for lib in libs.borrow().iter() {
            if relevant_lib(self.sess, lib) && lib.kind == kind {
                items.extend(&lib.foreign_items);
            }
        }
        items
    }

    fn register_statically_included_foreign_items(&mut self) {
        for id in self.get_foreign_items_of_kind(cstore::NativeStatic) {
            self.cstore.add_statically_included_foreign_item(id);
        }
    }

    fn register_dllimport_foreign_items(&mut self) {
        let mut dllimports = self.cstore.dllimport_foreign_items.borrow_mut();
        for id in self.get_foreign_items_of_kind(cstore::NativeUnknown) {
            dllimports.insert(id);
        }
    }

    fn inject_panic_runtime(&mut self, krate: &ast::Crate) {
        // If we're only compiling an rlib, then there's no need to select a
        // panic runtime, so we just skip this section entirely.
        let any_non_rlib = self.sess.crate_types.borrow().iter().any(|ct| {
            *ct != config::CrateTypeRlib
        });
        if !any_non_rlib {
            info!("panic runtime injection skipped, only generating rlib");
            return
        }

        // If we need a panic runtime, we try to find an existing one here. At
        // the same time we perform some general validation of the DAG we've got
        // going such as ensuring everything has a compatible panic strategy.
        //
        // The logic for finding the panic runtime here is pretty much the same
        // as the allocator case with the only addition that the panic strategy
        // compilation mode also comes into play.
        let desired_strategy = self.sess.panic_strategy();
        let mut runtime_found = false;
        let mut needs_panic_runtime = attr::contains_name(&krate.attrs,
                                                          "needs_panic_runtime");
        self.cstore.iter_crate_data(|cnum, data| {
            needs_panic_runtime = needs_panic_runtime || data.needs_panic_runtime();
            if data.is_panic_runtime() {
                // Inject a dependency from all #![needs_panic_runtime] to this
                // #![panic_runtime] crate.
                self.inject_dependency_if(cnum, "a panic runtime",
                                          &|data| data.needs_panic_runtime());
                runtime_found = runtime_found || data.dep_kind.get() == DepKind::Explicit;
            }
        });

        // If an explicitly linked and matching panic runtime was found, or if
        // we just don't need one at all, then we're done here and there's
        // nothing else to do.
        if !needs_panic_runtime || runtime_found {
            return
        }

        // By this point we know that we (a) need a panic runtime and (b) no
        // panic runtime was explicitly linked. Here we just load an appropriate
        // default runtime for our panic strategy and then inject the
        // dependencies.
        //
        // We may resolve to an already loaded crate (as the crate may not have
        // been explicitly linked prior to this) and we may re-inject
        // dependencies again, but both of those situations are fine.
        //
        // Also note that we have yet to perform validation of the crate graph
        // in terms of everyone has a compatible panic runtime format, that's
        // performed later as part of the `dependency_format` module.
        let name = match desired_strategy {
            PanicStrategy::Unwind => Symbol::intern("panic_unwind"),
            PanicStrategy::Abort => Symbol::intern("panic_abort"),
        };
        info!("panic runtime not found -- loading {}", name);

        let dep_kind = DepKind::Implicit;
        let (cnum, data) =
            self.resolve_crate(&None, name, name, None, DUMMY_SP, PathKind::Crate, dep_kind);

        // Sanity check the loaded crate to ensure it is indeed a panic runtime
        // and the panic strategy is indeed what we thought it was.
        if !data.is_panic_runtime() {
            self.sess.err(&format!("the crate `{}` is not a panic runtime",
                                   name));
        }
        if data.panic_strategy() != desired_strategy {
            self.sess.err(&format!("the crate `{}` does not have the panic \
                                    strategy `{}`",
                                   name, desired_strategy.desc()));
        }

        self.sess.injected_panic_runtime.set(Some(cnum));
        self.inject_dependency_if(cnum, "a panic runtime",
                                  &|data| data.needs_panic_runtime());
    }

    fn inject_allocator_crate(&mut self) {
        // Make sure that we actually need an allocator, if none of our
        // dependencies need one then we definitely don't!
        //
        // Also, if one of our dependencies has an explicit allocator, then we
        // also bail out as we don't need to implicitly inject one.
        let mut needs_allocator = false;
        let mut found_required_allocator = false;
        self.cstore.iter_crate_data(|cnum, data| {
            needs_allocator = needs_allocator || data.needs_allocator();
            if data.is_allocator() {
                info!("{} required by rlib and is an allocator", data.name());
                self.inject_dependency_if(cnum, "an allocator",
                                          &|data| data.needs_allocator());
                found_required_allocator = found_required_allocator ||
                    data.dep_kind.get() == DepKind::Explicit;
            }
        });
        if !needs_allocator || found_required_allocator { return }

        // At this point we've determined that we need an allocator and no
        // previous allocator has been activated. We look through our outputs of
        // crate types to see what kind of allocator types we may need.
        //
        // The main special output type here is that rlibs do **not** need an
        // allocator linked in (they're just object files), only final products
        // (exes, dylibs, staticlibs) need allocators.
        let mut need_lib_alloc = false;
        let mut need_exe_alloc = false;
        for ct in self.sess.crate_types.borrow().iter() {
            match *ct {
                config::CrateTypeExecutable => need_exe_alloc = true,
                config::CrateTypeDylib |
                config::CrateTypeProcMacro |
                config::CrateTypeCdylib |
                config::CrateTypeStaticlib => need_lib_alloc = true,
                config::CrateTypeRlib => {}
            }
        }
        if !need_lib_alloc && !need_exe_alloc { return }

        // The default allocator crate comes from the custom target spec, and we
        // choose between the standard library allocator or exe allocator. This
        // distinction exists because the default allocator for binaries (where
        // the world is Rust) is different than library (where the world is
        // likely *not* Rust).
        //
        // If a library is being produced, but we're also flagged with `-C
        // prefer-dynamic`, then we interpret this as a *Rust* dynamic library
        // is being produced so we use the exe allocator instead.
        //
        // What this boils down to is:
        //
        // * Binaries use jemalloc
        // * Staticlibs and Rust dylibs use system malloc
        // * Rust dylibs used as dependencies to rust use jemalloc
        let name = if need_lib_alloc && !self.sess.opts.cg.prefer_dynamic {
            Symbol::intern(&self.sess.target.target.options.lib_allocation_crate)
        } else {
            Symbol::intern(&self.sess.target.target.options.exe_allocation_crate)
        };
        let dep_kind = DepKind::Implicit;
        let (cnum, data) =
            self.resolve_crate(&None, name, name, None, DUMMY_SP, PathKind::Crate, dep_kind);

        // Sanity check the crate we loaded to ensure that it is indeed an
        // allocator.
        if !data.is_allocator() {
            self.sess.err(&format!("the allocator crate `{}` is not tagged \
                                    with #![allocator]", data.name()));
        }

        self.sess.injected_allocator.set(Some(cnum));
        self.inject_dependency_if(cnum, "an allocator",
                                  &|data| data.needs_allocator());
    }

    fn inject_dependency_if(&self,
                            krate: CrateNum,
                            what: &str,
                            needs_dep: &Fn(&cstore::CrateMetadata) -> bool) {
        // don't perform this validation if the session has errors, as one of
        // those errors may indicate a circular dependency which could cause
        // this to stack overflow.
        if self.sess.has_errors() {
            return
        }

        // Before we inject any dependencies, make sure we don't inject a
        // circular dependency by validating that this crate doesn't
        // transitively depend on any crates satisfying `needs_dep`.
        for dep in self.cstore.crate_dependencies_in_rpo(krate) {
            let data = self.cstore.get_crate_data(dep);
            if needs_dep(&data) {
                self.sess.err(&format!("the crate `{}` cannot depend \
                                        on a crate that needs {}, but \
                                        it depends on `{}`",
                                       self.cstore.get_crate_data(krate).name(),
                                       what,
                                       data.name()));
            }
        }

        // All crates satisfying `needs_dep` do not explicitly depend on the
        // crate provided for this compile, but in order for this compilation to
        // be successfully linked we need to inject a dependency (to order the
        // crates on the command line correctly).
        self.cstore.iter_crate_data(|cnum, data| {
            if !needs_dep(data) {
                return
            }

            info!("injecting a dep from {} to {}", cnum, krate);
            data.cnum_map.borrow_mut().push(krate);
        });
    }
}

impl<'a> CrateLoader<'a> {
    pub fn preprocess(&mut self, krate: &ast::Crate) {
        for attr in krate.attrs.iter().filter(|m| m.name() == "link_args") {
            if let Some(linkarg) = attr.value_str() {
                self.cstore.add_used_link_args(&linkarg.as_str());
            }
        }
    }

    fn process_foreign_mod(&mut self, i: &ast::Item, fm: &ast::ForeignMod,
                           definitions: &Definitions) {
        if fm.abi == Abi::Rust || fm.abi == Abi::RustIntrinsic || fm.abi == Abi::PlatformIntrinsic {
            return;
        }

        // First, add all of the custom #[link_args] attributes
        for m in i.attrs.iter().filter(|a| a.check_name("link_args")) {
            if let Some(linkarg) = m.value_str() {
                self.cstore.add_used_link_args(&linkarg.as_str());
            }
        }

        // Next, process all of the #[link(..)]-style arguments
        for m in i.attrs.iter().filter(|a| a.check_name("link")) {
            let items = match m.meta_item_list() {
                Some(item) => item,
                None => continue,
            };
            let kind = items.iter().find(|k| {
                k.check_name("kind")
            }).and_then(|a| a.value_str()).map(Symbol::as_str);
            let kind = match kind.as_ref().map(|s| &s[..]) {
                Some("static") => cstore::NativeStatic,
                Some("dylib") => cstore::NativeUnknown,
                Some("framework") => cstore::NativeFramework,
                Some(k) => {
                    struct_span_err!(self.sess, m.span, E0458,
                              "unknown kind: `{}`", k)
                        .span_label(m.span, &format!("unknown kind")).emit();
                    cstore::NativeUnknown
                }
                None => cstore::NativeUnknown
            };
            let n = items.iter().find(|n| {
                n.check_name("name")
            }).and_then(|a| a.value_str());
            let n = match n {
                Some(n) => n,
                None => {
                    struct_span_err!(self.sess, m.span, E0459,
                                     "#[link(...)] specified without `name = \"foo\"`")
                        .span_label(m.span, &format!("missing `name` argument")).emit();
                    Symbol::intern("foo")
                }
            };
            let cfg = items.iter().find(|k| {
                k.check_name("cfg")
            }).and_then(|a| a.meta_item_list());
            let cfg = cfg.map(|list| {
                list[0].meta_item().unwrap().clone()
            });
            let foreign_items = fm.items.iter()
                .map(|it| definitions.opt_def_index(it.id).unwrap())
                .collect();
            let lib = NativeLibrary {
                name: n,
                kind: kind,
                cfg: cfg,
                foreign_items: foreign_items,
            };
            register_native_lib(self.sess, self.cstore, Some(m.span), lib);
        }
    }
}

impl<'a> middle::cstore::CrateLoader for CrateLoader<'a> {
    fn postprocess(&mut self, krate: &ast::Crate) {
        self.inject_allocator_crate();
        self.inject_panic_runtime(krate);

        if log_enabled!(log::INFO) {
            dump_crates(&self.cstore);
        }

        // Process libs passed on the command line
        // First, check for errors
        let mut renames = FxHashSet();
        for &(ref name, ref new_name, _) in &self.sess.opts.libs {
            if let &Some(ref new_name) = new_name {
                if new_name.is_empty() {
                    self.sess.err(
                        &format!("an empty renaming target was specified for library `{}`",name));
                } else if !self.cstore.get_used_libraries().borrow().iter()
                                                           .any(|lib| lib.name == name as &str) {
                    self.sess.err(&format!("renaming of the library `{}` was specified, \
                                            however this crate contains no #[link(...)] \
                                            attributes referencing this library.", name));
                } else if renames.contains(name) {
                    self.sess.err(&format!("multiple renamings were specified for library `{}` .",
                                            name));
                } else {
                    renames.insert(name);
                }
            }
        }
        // Update kind and, optionally, the name of all native libaries
        // (there may be more than one) with the specified name.
        for &(ref name, ref new_name, kind) in &self.sess.opts.libs {
            let mut found = false;
            for lib in self.cstore.get_used_libraries().borrow_mut().iter_mut() {
                if lib.name == name as &str {
                    lib.kind = kind;
                    if let &Some(ref new_name) = new_name {
                        lib.name = Symbol::intern(new_name);
                    }
                    found = true;
                }
            }
            if !found {
                // Add if not found
                let new_name = new_name.as_ref().map(|s| &**s); // &Option<String> -> Option<&str>
                let lib = NativeLibrary {
                    name: Symbol::intern(new_name.unwrap_or(name)),
                    kind: kind,
                    cfg: None,
                    foreign_items: Vec::new(),
                };
                register_native_lib(self.sess, self.cstore, None, lib);
            }
        }
        self.register_statically_included_foreign_items();
        self.register_dllimport_foreign_items();
    }

    fn process_item(&mut self, item: &ast::Item, definitions: &Definitions) {
        match item.node {
            ast::ItemKind::ForeignMod(ref fm) => {
                self.process_foreign_mod(item, fm, definitions)
            },
            ast::ItemKind::ExternCrate(_) => {
                let info = self.extract_crate_info(item).unwrap();
                let (cnum, ..) = self.resolve_crate(
                    &None, info.ident, info.name, None, item.span, PathKind::Crate, info.dep_kind,
                );

                let def_id = definitions.opt_local_def_id(item.id).unwrap();
                let len = definitions.def_path(def_id.index).data.len();

                let extern_crate =
                    ExternCrate { def_id: def_id, span: item.span, direct: true, path_len: len };
                self.update_extern_crate(cnum, extern_crate, &mut FxHashSet());
                self.cstore.add_extern_mod_stmt_cnum(info.id, cnum);
            }
            _ => {}
        }
    }
}
