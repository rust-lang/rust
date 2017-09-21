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
use native_libs::relevant_lib;
use schema::CrateRoot;

use rustc::hir::def_id::{CrateNum, DefIndex, CRATE_DEF_INDEX};
use rustc::hir::svh::Svh;
use rustc::middle::allocator::AllocatorKind;
use rustc::middle::cstore::DepKind;
use rustc::session::Session;
use rustc::session::config::{Sanitizer, self};
use rustc_back::PanicStrategy;
use rustc::session::search_paths::PathKind;
use rustc::middle;
use rustc::middle::cstore::{validate_crate_name, ExternCrate};
use rustc::util::common::record_time;
use rustc::util::nodemap::FxHashSet;
use rustc::hir::map::Definitions;

use std::cell::{RefCell, Cell};
use std::ops::Deref;
use std::path::PathBuf;
use std::rc::Rc;
use std::{cmp, fs};

use syntax::ast;
use syntax::attr;
use syntax::ext::base::SyntaxExtension;
use syntax::symbol::Symbol;
use syntax::visit;
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
            sess,
            cstore,
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
                    name,
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
            let source = &self.cstore.get_crate_data(cnum).source;
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
                                  .or(source.rmeta.as_ref())
                                  .expect("No sources for crate").1;
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

        let trait_impls = crate_root
            .impls
            .decode(&metadata)
            .map(|trait_impls| (trait_impls.trait_id, trait_impls.impls))
            .collect();

        let mut cmeta = cstore::CrateMetadata {
            name,
            extern_crate: Cell::new(None),
            def_path_table: Rc::new(def_path_table),
            exported_symbols,
            trait_impls,
            proc_macros: crate_root.macro_derive_registrar.map(|_| {
                self.load_derive_macros(&crate_root, dylib.clone().map(|p| p.0), span)
            }),
            root: crate_root,
            blob: metadata,
            cnum_map: RefCell::new(cnum_map),
            cnum,
            codemap_import_info: RefCell::new(vec![]),
            attribute_cache: RefCell::new([Vec::new(), Vec::new()]),
            dep_kind: Cell::new(dep_kind),
            source: cstore::CrateSource {
                dylib,
                rlib,
                rmeta,
            },
            // Initialize this with an empty set. The field is populated below
            // after we were able to deserialize its contents.
            dllimport_foreign_items: FxHashSet(),
        };

        let dllimports: FxHashSet<_> = cmeta
            .root
            .native_libraries
            .decode(&cmeta)
            .filter(|lib| relevant_lib(self.sess, lib) &&
                          lib.kind == cstore::NativeLibraryKind::NativeUnknown)
            .flat_map(|lib| {
                assert!(lib.foreign_items.iter().all(|def_id| def_id.krate == cnum));
                lib.foreign_items.into_iter().map(|def_id| def_id.index)
            })
            .collect();

        cmeta.dllimport_foreign_items = dllimports;

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
                span,
                ident,
                crate_name: name,
                hash: hash.map(|a| &*a),
                filesearch: self.sess.target_filesearch(path_kind),
                target: &self.sess.target.target,
                triple: &self.sess.opts.target_triple,
                root,
                rejected_via_hash: vec![],
                rejected_via_triple: vec![],
                rejected_via_kind: vec![],
                rejected_via_version: vec![],
                rejected_via_filename: vec![],
                should_match_name: true,
                is_proc_macro: Some(false),
                metadata_loader: &*self.cstore.metadata_loader,
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
        ::std::iter::once(krate).chain(crate_root.crate_deps
                                                 .decode(metadata)
                                                 .map(|dep| {
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
            span,
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
            metadata_loader: &*self.cstore.metadata_loader,
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
            metadata,
            dylib: dylib.map(|p| p.0),
            target_only,
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
        use syntax_ext::deriving::custom::ProcMacroDerive;
        use syntax_ext::proc_macro_impl::{AttrProcMacro, BangProcMacro};

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

        let sym = self.sess.generate_derive_registrar_symbol(root.disambiguator,
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
                let attrs = attributes.iter().cloned().map(Symbol::intern).collect::<Vec<_>>();
                let derive = ProcMacroDerive::new(expand, attrs.clone());
                let derive = SyntaxExtension::ProcMacroDerive(Box::new(derive), attrs);
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

            fn register_bang_proc_macro(&mut self,
                                        name: &str,
                                        expand: fn(TokenStream) -> TokenStream) {
                let expand = SyntaxExtension::ProcMacro(
                    Box::new(BangProcMacro { inner: expand })
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
    pub fn find_plugin_registrar(&mut self,
                                 span: Span,
                                 name: &str)
                                 -> Option<(PathBuf, Symbol, DefIndex)> {
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
            span_fatal!(self.sess, span, E0456, "{}", &message);
        }

        let root = ekrate.metadata.get_root();
        match (ekrate.dylib.as_ref(), root.plugin_registrar_fn) {
            (Some(dylib), Some(reg)) => {
                Some((dylib.to_path_buf(), root.disambiguator, reg))
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
            needs_panic_runtime = needs_panic_runtime ||
                                  data.needs_panic_runtime();
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

    fn inject_sanitizer_runtime(&mut self) {
        if let Some(ref sanitizer) = self.sess.opts.debugging_opts.sanitizer {
            // Sanitizers can only be used on some tested platforms with
            // executables linked to `std`
            const ASAN_SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu",
                                                      "x86_64-apple-darwin"];
            const TSAN_SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu",
                                                      "x86_64-apple-darwin"];
            const LSAN_SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu"];
            const MSAN_SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu"];

            let supported_targets = match *sanitizer {
                Sanitizer::Address => ASAN_SUPPORTED_TARGETS,
                Sanitizer::Thread => TSAN_SUPPORTED_TARGETS,
                Sanitizer::Leak => LSAN_SUPPORTED_TARGETS,
                Sanitizer::Memory => MSAN_SUPPORTED_TARGETS,
            };
            if !supported_targets.contains(&&*self.sess.target.target.llvm_target) {
                self.sess.err(&format!("{:?}Sanitizer only works with the `{}` target",
                    sanitizer,
                    supported_targets.join("` or `")
                ));
                return
            }

            // firstyear 2017 - during testing I was unable to access an OSX machine
            // to make this work on different crate types. As a result, today I have
            // only been able to test and support linux as a target.
            if self.sess.target.target.llvm_target == "x86_64-unknown-linux-gnu" {
                if !self.sess.crate_types.borrow().iter().all(|ct| {
                    match *ct {
                        // Link the runtime
                        config::CrateTypeStaticlib |
                        config::CrateTypeExecutable => true,
                        // This crate will be compiled with the required
                        // instrumentation pass
                        config::CrateTypeRlib |
                        config::CrateTypeDylib |
                        config::CrateTypeCdylib =>
                            false,
                        _ => {
                            self.sess.err(&format!("Only executables, staticlibs, \
                                cdylibs, dylibs and rlibs can be compiled with \
                                `-Z sanitizer`"));
                            false
                        }
                    }
                }) {
                    return
                }
            } else {
                if !self.sess.crate_types.borrow().iter().all(|ct| {
                    match *ct {
                        // Link the runtime
                        config::CrateTypeExecutable => true,
                        // This crate will be compiled with the required
                        // instrumentation pass
                        config::CrateTypeRlib => false,
                        _ => {
                            self.sess.err(&format!("Only executables and rlibs can be \
                                                    compiled with `-Z sanitizer`"));
                            false
                        }
                    }
                }) {
                    return
                }
            }

            let mut uses_std = false;
            self.cstore.iter_crate_data(|_, data| {
                if data.name == "std" {
                    uses_std = true;
                }
            });

            if uses_std {
                let name = match *sanitizer {
                    Sanitizer::Address => "rustc_asan",
                    Sanitizer::Leak => "rustc_lsan",
                    Sanitizer::Memory => "rustc_msan",
                    Sanitizer::Thread => "rustc_tsan",
                };
                info!("loading sanitizer: {}", name);

                let symbol = Symbol::intern(name);
                let dep_kind = DepKind::Explicit;
                let (_, data) =
                    self.resolve_crate(&None, symbol, symbol, None, DUMMY_SP,
                                       PathKind::Crate, dep_kind);

                // Sanity check the loaded crate to ensure it is indeed a sanitizer runtime
                if !data.is_sanitizer_runtime() {
                    self.sess.err(&format!("the crate `{}` is not a sanitizer runtime",
                                           name));
                }
            } else {
                self.sess.err(&format!("Must link std to be compiled with `-Z sanitizer`"));
            }
        }
    }

    fn inject_profiler_runtime(&mut self) {
        if self.sess.opts.debugging_opts.profile {
            info!("loading profiler");

            let symbol = Symbol::intern("profiler_builtins");
            let dep_kind = DepKind::Implicit;
            let (_, data) =
                self.resolve_crate(&None, symbol, symbol, None, DUMMY_SP,
                                   PathKind::Crate, dep_kind);

            // Sanity check the loaded crate to ensure it is indeed a profiler runtime
            if !data.is_profiler_runtime() {
                self.sess.err(&format!("the crate `profiler_builtins` is not \
                                        a profiler runtime"));
            }
        }
    }

    fn inject_allocator_crate(&mut self, krate: &ast::Crate) {
        let has_global_allocator = has_global_allocator(krate);
        if has_global_allocator {
            self.sess.has_global_allocator.set(true);
        }

        // Check to see if we actually need an allocator. This desire comes
        // about through the `#![needs_allocator]` attribute and is typically
        // written down in liballoc.
        let mut needs_allocator = attr::contains_name(&krate.attrs,
                                                      "needs_allocator");
        self.cstore.iter_crate_data(|_, data| {
            needs_allocator = needs_allocator || data.needs_allocator();
        });
        if !needs_allocator {
            return
        }

        // At this point we've determined that we need an allocator. Let's see
        // if our compilation session actually needs an allocator based on what
        // we're emitting.
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
        if !need_lib_alloc && !need_exe_alloc {
            return
        }

        // Ok, we need an allocator. Not only that but we're actually going to
        // create an artifact that needs one linked in. Let's go find the one
        // that we're going to link in.
        //
        // First up we check for global allocators. Look at the crate graph here
        // and see what's a global allocator, including if we ourselves are a
        // global allocator.
        let mut global_allocator = if has_global_allocator {
            Some(None)
        } else {
            None
        };
        self.cstore.iter_crate_data(|_, data| {
            if !data.has_global_allocator() {
                return
            }
            match global_allocator {
                Some(Some(other_crate)) => {
                    self.sess.err(&format!("the #[global_allocator] in {} \
                                            conflicts with this global \
                                            allocator in: {}",
                                           other_crate,
                                           data.name()));
                }
                Some(None) => {
                    self.sess.err(&format!("the #[global_allocator] in this \
                                            crate conflicts with global \
                                            allocator in: {}", data.name()));
                }
                None => global_allocator = Some(Some(data.name())),
            }
        });
        if global_allocator.is_some() {
            self.sess.allocator_kind.set(Some(AllocatorKind::Global));
            return
        }

        // Ok we haven't found a global allocator but we still need an
        // allocator. At this point we'll either fall back to the "library
        // allocator" or the "exe allocator" depending on a few variables. Let's
        // figure out which one.
        //
        // Note that here we favor linking to the "library allocator" as much as
        // possible. If we're not creating rustc's version of libstd
        // (need_lib_alloc and prefer_dynamic) then we select `None`, and if the
        // exe allocation crate doesn't exist for this target then we also
        // select `None`.
        let exe_allocation_crate_data =
            if need_lib_alloc && !self.sess.opts.cg.prefer_dynamic {
                None
            } else {
                self.sess
                    .target
                    .target
                    .options
                    .exe_allocation_crate
                    .as_ref()
                    .map(|name| {
                        // We've determined that we're injecting an "exe allocator" which means
                        // that we're going to load up a whole new crate. An example of this is
                        // that we're producing a normal binary on Linux which means we need to
                        // load the `alloc_jemalloc` crate to link as an allocator.
                        let name = Symbol::intern(name);
                        let (cnum, data) = self.resolve_crate(&None,
                                                              name,
                                                              name,
                                                              None,
                                                              DUMMY_SP,
                                                              PathKind::Crate,
                                                              DepKind::Implicit);
                        self.sess.injected_allocator.set(Some(cnum));
                        data
                    })
            };

        let allocation_crate_data = exe_allocation_crate_data.or_else(|| {
            if attr::contains_name(&krate.attrs, "default_lib_allocator") {
                // Prefer self as the allocator if there's a collision
                return None;
            }
            // We're not actually going to inject an allocator, we're going to
            // require that something in our crate graph is the default lib
            // allocator. This is typically libstd, so this'll rarely be an
            // error.
            let mut allocator = None;
            self.cstore.iter_crate_data(|_, data| {
                if allocator.is_none() && data.has_default_lib_allocator() {
                    allocator = Some(data.clone());
                }
            });
            allocator
        });

        match allocation_crate_data {
            Some(data) => {
                // We have an allocator. We detect separately what kind it is, to allow for some
                // flexibility in misconfiguration.
                let attrs = data.get_item_attrs(CRATE_DEF_INDEX);
                let kind_interned = attr::first_attr_value_str_by_name(&attrs, "rustc_alloc_kind")
                    .map(Symbol::as_str);
                let kind_str = kind_interned
                    .as_ref()
                    .map(|s| s as &str);
                let alloc_kind = match kind_str {
                    None |
                    Some("lib") => AllocatorKind::DefaultLib,
                    Some("exe") => AllocatorKind::DefaultExe,
                    Some(other) => {
                        self.sess.err(&format!("Allocator kind {} not known", other));
                        return;
                    }
                };
                self.sess.allocator_kind.set(Some(alloc_kind));
            },
            None => {
                if !attr::contains_name(&krate.attrs, "default_lib_allocator") {
                    self.sess.err("no #[default_lib_allocator] found but one is \
                                   required; is libstd not linked?");
                    return;
                }
                self.sess.allocator_kind.set(Some(AllocatorKind::DefaultLib));
            }
        }

        fn has_global_allocator(krate: &ast::Crate) -> bool {
            struct Finder(bool);
            let mut f = Finder(false);
            visit::walk_crate(&mut f, krate);
            return f.0;

            impl<'ast> visit::Visitor<'ast> for Finder {
                fn visit_item(&mut self, i: &'ast ast::Item) {
                    if attr::contains_name(&i.attrs, "global_allocator") {
                        self.0 = true;
                    }
                    visit::walk_item(self, i)
                }
            }
        }
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

impl<'a> middle::cstore::CrateLoader for CrateLoader<'a> {
    fn postprocess(&mut self, krate: &ast::Crate) {
        // inject the sanitizer runtime before the allocator runtime because all
        // sanitizers force the use of the `alloc_system` allocator
        self.inject_sanitizer_runtime();
        self.inject_profiler_runtime();
        self.inject_allocator_crate(krate);
        self.inject_panic_runtime(krate);

        if log_enabled!(log::LogLevel::Info) {
            dump_crates(&self.cstore);
        }
    }

    fn process_item(&mut self, item: &ast::Item, definitions: &Definitions) {
        match item.node {
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
