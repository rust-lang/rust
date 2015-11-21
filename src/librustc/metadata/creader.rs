// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

//! Validates all used crates and extern libraries and loads their metadata

use back::svh::Svh;
use session::{config, Session};
use session::search_paths::PathKind;
use metadata::common::rustc_version;
use metadata::cstore;
use metadata::cstore::{CStore, CrateSource, MetadataBlob};
use metadata::decoder;
use metadata::loader;
use metadata::loader::CratePaths;
use util::nodemap::FnvHashMap;
use front::map as hir_map;

use std::cell::{RefCell, Cell};
use std::path::PathBuf;
use std::rc::Rc;
use std::fs;

use syntax::ast;
use syntax::abi;
use syntax::codemap::{self, Span, mk_sp, Pos};
use syntax::parse;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::InternedString;
use syntax::util::small_vector::SmallVector;
use rustc_front::intravisit::Visitor;
use rustc_front::hir;
use log;

pub struct LocalCrateReader<'a, 'b:'a> {
    sess: &'a Session,
    cstore: &'a CStore,
    creader: CrateReader<'a>,
    ast_map: &'a hir_map::Map<'b>,
}

pub struct CrateReader<'a> {
    sess: &'a Session,
    cstore: &'a CStore,
    next_crate_num: ast::CrateNum,
    foreign_item_map: FnvHashMap<String, Vec<ast::NodeId>>,
}

impl<'a, 'b, 'hir> Visitor<'hir> for LocalCrateReader<'a, 'b> {
    fn visit_item(&mut self, a: &'hir hir::Item) {
        self.process_item(a);
    }
}

fn dump_crates(cstore: &CStore) {
    info!("resolved crates:");
    cstore.iter_crate_data_origins(|_, data, opt_source| {
        info!("  name: {}", data.name());
        info!("  cnum: {}", data.cnum);
        info!("  hash: {}", data.hash());
        info!("  reqd: {}", data.explicitly_linked.get());
        opt_source.map(|cs| {
            let CrateSource { dylib, rlib, cnum: _ } = cs;
            dylib.map(|dl| info!("  dylib: {}", dl.0.display()));
            rlib.map(|rl|  info!("   rlib: {}", rl.0.display()));
        });
    })
}

fn should_link(i: &ast::Item) -> bool {
    !attr::contains_name(&i.attrs, "no_link")
}
// Dup for the hir
fn should_link_hir(i: &hir::Item) -> bool {
    !attr::contains_name(&i.attrs, "no_link")
}

struct CrateInfo {
    ident: String,
    name: String,
    id: ast::NodeId,
    should_link: bool,
}

pub fn validate_crate_name(sess: Option<&Session>, s: &str, sp: Option<Span>) {
    let say = |s: &str| {
        match (sp, sess) {
            (_, None) => panic!("{}", s),
            (Some(sp), Some(sess)) => sess.span_err(sp, s),
            (None, Some(sess)) => sess.err(s),
        }
    };
    if s.is_empty() {
        say("crate name must not be empty");
    }
    for c in s.chars() {
        if c.is_alphanumeric() { continue }
        if c == '_'  { continue }
        say(&format!("invalid character `{}` in crate name: `{}`", c, s));
    }
    match sess {
        Some(sess) => sess.abort_if_errors(),
        None => {}
    }
}


fn register_native_lib(sess: &Session,
                       cstore: &CStore,
                       span: Option<Span>,
                       name: String,
                       kind: cstore::NativeLibraryKind) {
    if name.is_empty() {
        match span {
            Some(span) => {
                span_err!(sess, span, E0454,
                          "#[link(name = \"\")] given with empty name");
            }
            None => {
                sess.err("empty library name given via `-l`");
            }
        }
        return
    }
    let is_osx = sess.target.target.options.is_like_osx;
    if kind == cstore::NativeFramework && !is_osx {
        let msg = "native frameworks are only available on OSX targets";
        match span {
            Some(span) => {
                span_err!(sess, span, E0455,
                          "{}", msg)
            }
            None => sess.err(msg),
        }
    }
    cstore.add_used_library(name, kind);
}

// Extra info about a crate loaded for plugins or exported macros.
struct ExtensionCrate {
    metadata: PMDSource,
    dylib: Option<PathBuf>,
    target_only: bool,
}

enum PMDSource {
    Registered(Rc<cstore::crate_metadata>),
    Owned(MetadataBlob),
}

impl PMDSource {
    pub fn as_slice<'a>(&'a self) -> &'a [u8] {
        match *self {
            PMDSource::Registered(ref cmd) => cmd.data(),
            PMDSource::Owned(ref mdb) => mdb.as_slice(),
        }
    }
}

impl<'a> CrateReader<'a> {
    pub fn new(sess: &'a Session, cstore: &'a CStore) -> CrateReader<'a> {
        CrateReader {
            sess: sess,
            cstore: cstore,
            next_crate_num: cstore.next_crate_num(),
            foreign_item_map: FnvHashMap(),
        }
    }

    fn extract_crate_info(&self, i: &ast::Item) -> Option<CrateInfo> {
        match i.node {
            ast::ItemExternCrate(ref path_opt) => {
                debug!("resolving extern crate stmt. ident: {} path_opt: {:?}",
                       i.ident, path_opt);
                let name = match *path_opt {
                    Some(name) => {
                        validate_crate_name(Some(self.sess), &name.as_str(),
                                            Some(i.span));
                        name.to_string()
                    }
                    None => i.ident.to_string(),
                };
                Some(CrateInfo {
                    ident: i.ident.to_string(),
                    name: name,
                    id: i.id,
                    should_link: should_link(i),
                })
            }
            _ => None
        }
    }

    // Dup of the above, but for the hir
    fn extract_crate_info_hir(&self, i: &hir::Item) -> Option<CrateInfo> {
        match i.node {
            hir::ItemExternCrate(ref path_opt) => {
                debug!("resolving extern crate stmt. ident: {} path_opt: {:?}",
                       i.name, path_opt);
                let name = match *path_opt {
                    Some(name) => {
                        validate_crate_name(Some(self.sess), &name.as_str(),
                                            Some(i.span));
                        name.to_string()
                    }
                    None => i.name.to_string(),
                };
                Some(CrateInfo {
                    ident: i.name.to_string(),
                    name: name,
                    id: i.id,
                    should_link: should_link_hir(i),
                })
            }
            _ => None
        }
    }

    fn existing_match(&self, name: &str, hash: Option<&Svh>, kind: PathKind)
                      -> Option<ast::CrateNum> {
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
            let source = self.cstore.do_get_used_crate_source(cnum).unwrap();
            if let Some(locs) = self.sess.opts.externs.get(name) {
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

    fn verify_rustc_version(&self,
                            name: &str,
                            span: Span,
                            metadata: &MetadataBlob) {
        let crate_rustc_version = decoder::crate_rustc_version(metadata.as_slice());
        if crate_rustc_version != Some(rustc_version()) {
            span_err!(self.sess, span, E0514,
                      "the crate `{}` has been compiled with {}, which is \
                       incompatible with this version of rustc",
                      name,
                      crate_rustc_version
                          .as_ref().map(|s|&**s)
                          .unwrap_or("an old version of rustc")
            );
            self.sess.abort_if_errors();
        }
    }

    fn register_crate(&mut self,
                      root: &Option<CratePaths>,
                      ident: &str,
                      name: &str,
                      span: Span,
                      lib: loader::Library,
                      explicitly_linked: bool)
                      -> (ast::CrateNum, Rc<cstore::crate_metadata>,
                          cstore::CrateSource) {
        self.verify_rustc_version(name, span, &lib.metadata);

        // Claim this crate number and cache it
        let cnum = self.next_crate_num;
        self.next_crate_num += 1;

        // Stash paths for top-most crate locally if necessary.
        let crate_paths = if root.is_none() {
            Some(CratePaths {
                ident: ident.to_string(),
                dylib: lib.dylib.clone().map(|p| p.0),
                rlib:  lib.rlib.clone().map(|p| p.0),
            })
        } else {
            None
        };
        // Maintain a reference to the top most crate.
        let root = if root.is_some() { root } else { &crate_paths };

        let loader::Library { dylib, rlib, metadata } = lib;

        let cnum_map = self.resolve_crate_deps(root, metadata.as_slice(), span);
        let staged_api = self.is_staged_api(metadata.as_slice());

        let cmeta = Rc::new(cstore::crate_metadata {
            name: name.to_string(),
            local_path: RefCell::new(SmallVector::zero()),
            local_def_path: RefCell::new(vec![]),
            index: decoder::load_index(metadata.as_slice()),
            xref_index: decoder::load_xrefs(metadata.as_slice()),
            data: metadata,
            cnum_map: RefCell::new(cnum_map),
            cnum: cnum,
            codemap_import_info: RefCell::new(vec![]),
            span: span,
            staged_api: staged_api,
            explicitly_linked: Cell::new(explicitly_linked),
        });

        let source = cstore::CrateSource {
            dylib: dylib,
            rlib: rlib,
            cnum: cnum,
        };

        self.cstore.set_crate_data(cnum, cmeta.clone());
        self.cstore.add_used_crate_source(source.clone());
        (cnum, cmeta, source)
    }

    fn is_staged_api(&self, data: &[u8]) -> bool {
        let attrs = decoder::get_crate_attributes(data);
        for attr in &attrs {
            if attr.name() == "stable" || attr.name() == "unstable" {
                return true
            }
        }
        false
    }

    fn resolve_crate(&mut self,
                     root: &Option<CratePaths>,
                     ident: &str,
                     name: &str,
                     hash: Option<&Svh>,
                     span: Span,
                     kind: PathKind,
                     explicitly_linked: bool)
                         -> (ast::CrateNum, Rc<cstore::crate_metadata>,
                             cstore::CrateSource) {
        enum LookupResult {
            Previous(ast::CrateNum),
            Loaded(loader::Library),
        }
        let result = match self.existing_match(name, hash, kind) {
            Some(cnum) => LookupResult::Previous(cnum),
            None => {
                let mut load_ctxt = loader::Context {
                    sess: self.sess,
                    span: span,
                    ident: ident,
                    crate_name: name,
                    hash: hash.map(|a| &*a),
                    filesearch: self.sess.target_filesearch(kind),
                    target: &self.sess.target.target,
                    triple: &self.sess.opts.target_triple,
                    root: root,
                    rejected_via_hash: vec!(),
                    rejected_via_triple: vec!(),
                    rejected_via_kind: vec!(),
                    should_match_name: true,
                };
                let library = load_ctxt.load_library_crate();

                // In the case that we're loading a crate, but not matching
                // against a hash, we could load a crate which has the same hash
                // as an already loaded crate. If this is the case prevent
                // duplicates by just using the first crate.
                let meta_hash = decoder::get_crate_hash(library.metadata
                                                               .as_slice());
                let mut result = LookupResult::Loaded(library);
                self.cstore.iter_crate_data(|cnum, data| {
                    if data.name() == name && meta_hash == data.hash() {
                        assert!(hash.is_none());
                        result = LookupResult::Previous(cnum);
                    }
                });
                result
            }
        };

        match result {
            LookupResult::Previous(cnum) => {
                let data = self.cstore.get_crate_data(cnum);
                if explicitly_linked && !data.explicitly_linked.get() {
                    data.explicitly_linked.set(explicitly_linked);
                }
                (cnum, data, self.cstore.do_get_used_crate_source(cnum).unwrap())
            }
            LookupResult::Loaded(library) => {
                self.register_crate(root, ident, name, span, library,
                                    explicitly_linked)
            }
        }
    }

    // Go through the crate metadata and load any crates that it references
    fn resolve_crate_deps(&mut self,
                          root: &Option<CratePaths>,
                          cdata: &[u8], span : Span)
                       -> cstore::cnum_map {
        debug!("resolving deps of external crate");
        // The map from crate numbers in the crate we're resolving to local crate
        // numbers
        decoder::get_crate_deps(cdata).iter().map(|dep| {
            debug!("resolving dep crate {} hash: `{}`", dep.name, dep.hash);
            let (local_cnum, _, _) = self.resolve_crate(root,
                                                   &dep.name,
                                                   &dep.name,
                                                   Some(&dep.hash),
                                                   span,
                                                   PathKind::Dependency,
                                                   dep.explicitly_linked);
            (dep.cnum, local_cnum)
        }).collect()
    }

    fn read_extension_crate(&mut self, span: Span, info: &CrateInfo) -> ExtensionCrate {
        let target_triple = &self.sess.opts.target_triple[..];
        let is_cross = target_triple != config::host_triple();
        let mut should_link = info.should_link && !is_cross;
        let mut target_only = false;
        let ident = info.ident.clone();
        let name = info.name.clone();
        let mut load_ctxt = loader::Context {
            sess: self.sess,
            span: span,
            ident: &ident[..],
            crate_name: &name[..],
            hash: None,
            filesearch: self.sess.host_filesearch(PathKind::Crate),
            target: &self.sess.host,
            triple: config::host_triple(),
            root: &None,
            rejected_via_hash: vec!(),
            rejected_via_triple: vec!(),
            rejected_via_kind: vec!(),
            should_match_name: true,
        };
        let library = match load_ctxt.maybe_load_library_crate() {
            Some(l) => l,
            None if is_cross => {
                // Try loading from target crates. This will abort later if we
                // try to load a plugin registrar function,
                target_only = true;
                should_link = info.should_link;

                load_ctxt.target = &self.sess.target.target;
                load_ctxt.triple = target_triple;
                load_ctxt.filesearch = self.sess.target_filesearch(PathKind::Crate);
                load_ctxt.load_library_crate()
            }
            None => { load_ctxt.report_load_errs(); unreachable!() },
        };

        let dylib = library.dylib.clone();
        let register = should_link && self.existing_match(&info.name,
                                                          None,
                                                          PathKind::Crate).is_none();
        let metadata = if register {
            // Register crate now to avoid double-reading metadata
            let (_, cmd, _) = self.register_crate(&None, &info.ident,
                                                  &info.name, span, library,
                                                  true);
            PMDSource::Registered(cmd)
        } else {
            // Not registering the crate; just hold on to the metadata
            PMDSource::Owned(library.metadata)
        };

        ExtensionCrate {
            metadata: metadata,
            dylib: dylib.map(|p| p.0),
            target_only: target_only,
        }
    }

    /// Read exported macros.
    pub fn read_exported_macros(&mut self, item: &ast::Item) -> Vec<ast::MacroDef> {
        let ci = self.extract_crate_info(item).unwrap();
        let ekrate = self.read_extension_crate(item.span, &ci);

        let source_name = format!("<{} macros>", item.ident);
        let mut macros = vec![];
        decoder::each_exported_macro(ekrate.metadata.as_slice(),
                                     &*self.cstore.intr,
            |name, attrs, body| {
                // NB: Don't use parse::parse_tts_from_source_str because it parses with
                // quote_depth > 0.
                let mut p = parse::new_parser_from_source_str(&self.sess.parse_sess,
                                                              self.sess.opts.cfg.clone(),
                                                              source_name.clone(),
                                                              body);
                let lo = p.span.lo;
                let body = match p.parse_all_token_trees() {
                    Ok(body) => body,
                    Err(err) => panic!(err),
                };
                let span = mk_sp(lo, p.last_span.hi);
                p.abort_if_errors();

                // Mark the attrs as used
                for attr in &attrs {
                    attr::mark_used(attr);
                }

                macros.push(ast::MacroDef {
                    ident: ast::Ident::with_empty_ctxt(name),
                    attrs: attrs,
                    id: ast::DUMMY_NODE_ID,
                    span: span,
                    imported_from: Some(item.ident),
                    // overridden in plugin/load.rs
                    export: false,
                    use_locally: false,
                    allow_internal_unstable: false,

                    body: body,
                });
                true
            }
        );
        macros
    }

    /// Look for a plugin registrar. Returns library path and symbol name.
    pub fn find_plugin_registrar(&mut self, span: Span, name: &str)
                                 -> Option<(PathBuf, String)> {
        let ekrate = self.read_extension_crate(span, &CrateInfo {
             name: name.to_string(),
             ident: name.to_string(),
             id: ast::DUMMY_NODE_ID,
             should_link: false,
        });

        if ekrate.target_only {
            // Need to abort before syntax expansion.
            let message = format!("plugin `{}` is not available for triple `{}` \
                                   (only found {})",
                                  name,
                                  config::host_triple(),
                                  self.sess.opts.target_triple);
            span_err!(self.sess, span, E0456, "{}", &message[..]);
            self.sess.abort_if_errors();
        }

        let registrar =
            decoder::get_plugin_registrar_fn(ekrate.metadata.as_slice())
            .map(|id| decoder::get_symbol_from_buf(ekrate.metadata.as_slice(), id));

        match (ekrate.dylib.as_ref(), registrar) {
            (Some(dylib), Some(reg)) => Some((dylib.to_path_buf(), reg)),
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

    fn register_statically_included_foreign_items(&mut self) {
        let libs = self.cstore.get_used_libraries();
        for (lib, list) in self.foreign_item_map.iter() {
            let is_static = libs.borrow().iter().any(|&(ref name, kind)| {
                lib == name && kind == cstore::NativeStatic
            });
            if is_static {
                for id in list {
                    self.cstore.add_statically_included_foreign_item(*id);
                }
            }
        }
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
                debug!("{} required by rlib and is an allocator", data.name());
                self.inject_allocator_dependency(cnum);
                found_required_allocator = found_required_allocator ||
                    data.explicitly_linked.get();
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
            &self.sess.target.target.options.lib_allocation_crate
        } else {
            &self.sess.target.target.options.exe_allocation_crate
        };
        let (cnum, data, _) = self.resolve_crate(&None, name, name, None,
                                                 codemap::DUMMY_SP,
                                                 PathKind::Crate, false);

        // To ensure that the `-Z allocation-crate=foo` option isn't abused, and
        // to ensure that the allocator is indeed an allocator, we verify that
        // the crate loaded here is indeed tagged #![allocator].
        if !data.is_allocator() {
            self.sess.err(&format!("the allocator crate `{}` is not tagged \
                                    with #![allocator]", data.name()));
        }

        self.sess.injected_allocator.set(Some(cnum));
        self.inject_allocator_dependency(cnum);
    }

    fn inject_allocator_dependency(&self, allocator: ast::CrateNum) {
        // Before we inject any dependencies, make sure we don't inject a
        // circular dependency by validating that this allocator crate doesn't
        // transitively depend on any `#![needs_allocator]` crates.
        validate(self, allocator, allocator);

        // All crates tagged with `needs_allocator` do not explicitly depend on
        // the allocator selected for this compile, but in order for this
        // compilation to be successfully linked we need to inject a dependency
        // (to order the crates on the command line correctly).
        //
        // Here we inject a dependency from all crates with #![needs_allocator]
        // to the crate tagged with #![allocator] for this compilation unit.
        self.cstore.iter_crate_data(|cnum, data| {
            if !data.needs_allocator() {
                return
            }

            info!("injecting a dep from {} to {}", cnum, allocator);
            let mut cnum_map = data.cnum_map.borrow_mut();
            let remote_cnum = cnum_map.len() + 1;
            let prev = cnum_map.insert(remote_cnum as ast::CrateNum, allocator);
            assert!(prev.is_none());
        });

        fn validate(me: &CrateReader, krate: ast::CrateNum,
                    allocator: ast::CrateNum) {
            let data = me.cstore.get_crate_data(krate);
            if data.needs_allocator() {
                let krate_name = data.name();
                let data = me.cstore.get_crate_data(allocator);
                let alloc_name = data.name();
                me.sess.err(&format!("the allocator crate `{}` cannot depend \
                                      on a crate that needs an allocator, but \
                                      it depends on `{}`", alloc_name,
                                      krate_name));
            }

            for (_, &dep) in data.cnum_map.borrow().iter() {
                validate(me, dep, allocator);
            }
        }
    }
}

impl<'a, 'b> LocalCrateReader<'a, 'b> {
    pub fn new(sess: &'a Session, cstore: &'a CStore, map: &'a hir_map::Map<'b>) -> LocalCrateReader<'a, 'b> {
        LocalCrateReader {
            sess: sess,
            cstore: cstore,
            creader: CrateReader::new(sess, cstore),
            ast_map: map,
        }
    }

    // Traverses an AST, reading all the information about use'd crates and
    // extern libraries necessary for later resolving, typechecking, linking,
    // etc.
    pub fn read_crates(&mut self, krate: &hir::Crate) {
        self.process_crate(krate);
        krate.visit_all_items(self);
        self.creader.inject_allocator_crate();

        if log_enabled!(log::INFO) {
            dump_crates(&self.cstore);
        }

        for &(ref name, kind) in &self.sess.opts.libs {
            register_native_lib(self.sess, self.cstore, None, name.clone(), kind);
        }
        self.creader.register_statically_included_foreign_items();
    }

    fn process_crate(&self, c: &hir::Crate) {
        for a in c.attrs.iter().filter(|m| m.name() == "link_args") {
            match a.value_str() {
                Some(ref linkarg) => self.cstore.add_used_link_args(&linkarg),
                None => { /* fallthrough */ }
            }
        }
    }

    fn process_item(&mut self, i: &hir::Item) {
        match i.node {
            hir::ItemExternCrate(_) => {
                if !should_link_hir(i) {
                    return;
                }

                match self.creader.extract_crate_info_hir(i) {
                    Some(info) => {
                        let (cnum, cmeta, _) = self.creader.resolve_crate(&None,
                                                              &info.ident,
                                                              &info.name,
                                                              None,
                                                              i.span,
                                                              PathKind::Crate,
                                                              true);
                        let def_id = self.ast_map.local_def_id(i.id);
                        let def_path = self.ast_map.def_path(def_id);
                        cmeta.update_local_def_path(def_path);
                        self.ast_map.with_path(i.id, |path| {
                            cmeta.update_local_path(path)
                        });
                        self.cstore.add_extern_mod_stmt_cnum(info.id, cnum);
                    }
                    None => ()
                }
            }
            hir::ItemForeignMod(ref fm) => self.process_foreign_mod(i, fm),
            _ => { }
        }
    }

    fn process_foreign_mod(&mut self, i: &hir::Item, fm: &hir::ForeignMod) {
        if fm.abi == abi::Rust || fm.abi == abi::RustIntrinsic || fm.abi == abi::PlatformIntrinsic {
            return;
        }

        // First, add all of the custom #[link_args] attributes
        for m in i.attrs.iter().filter(|a| a.check_name("link_args")) {
            if let Some(linkarg) = m.value_str() {
                self.cstore.add_used_link_args(&linkarg);
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
            }).and_then(|a| a.value_str());
            let kind = match kind.as_ref().map(|s| &s[..]) {
                Some("static") => cstore::NativeStatic,
                Some("dylib") => cstore::NativeUnknown,
                Some("framework") => cstore::NativeFramework,
                Some(k) => {
                    span_err!(self.sess, m.span, E0458,
                              "unknown kind: `{}`", k);
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
                    span_err!(self.sess, m.span, E0459,
                              "#[link(...)] specified without `name = \"foo\"`");
                    InternedString::new("foo")
                }
            };
            register_native_lib(self.sess, self.cstore, Some(m.span), n.to_string(), kind);
        }

        // Finally, process the #[linked_from = "..."] attribute
        for m in i.attrs.iter().filter(|a| a.check_name("linked_from")) {
            let lib_name = match m.value_str() {
                Some(name) => name,
                None => continue,
            };
            let list = self.creader.foreign_item_map.entry(lib_name.to_string())
                                                    .or_insert(Vec::new());
            list.extend(fm.items.iter().map(|it| it.id));
        }
    }
}

/// Imports the codemap from an external crate into the codemap of the crate
/// currently being compiled (the "local crate").
///
/// The import algorithm works analogous to how AST items are inlined from an
/// external crate's metadata:
/// For every FileMap in the external codemap an 'inline' copy is created in the
/// local codemap. The correspondence relation between external and local
/// FileMaps is recorded in the `ImportedFileMap` objects returned from this
/// function. When an item from an external crate is later inlined into this
/// crate, this correspondence information is used to translate the span
/// information of the inlined item so that it refers the correct positions in
/// the local codemap (see `astencode::DecodeContext::tr_span()`).
///
/// The import algorithm in the function below will reuse FileMaps already
/// existing in the local codemap. For example, even if the FileMap of some
/// source file of libstd gets imported many times, there will only ever be
/// one FileMap object for the corresponding file in the local codemap.
///
/// Note that imported FileMaps do not actually contain the source code of the
/// file they represent, just information about length, line breaks, and
/// multibyte characters. This information is enough to generate valid debuginfo
/// for items inlined from other crates.
pub fn import_codemap(local_codemap: &codemap::CodeMap,
                      metadata: &MetadataBlob)
                      -> Vec<cstore::ImportedFileMap> {
    let external_codemap = decoder::get_imported_filemaps(metadata.as_slice());

    let imported_filemaps = external_codemap.into_iter().map(|filemap_to_import| {
        // Try to find an existing FileMap that can be reused for the filemap to
        // be imported. A FileMap is reusable if it is exactly the same, just
        // positioned at a different offset within the codemap.
        let reusable_filemap = {
            local_codemap.files
                         .borrow()
                         .iter()
                         .find(|fm| are_equal_modulo_startpos(&fm, &filemap_to_import))
                         .map(|rc| rc.clone())
        };

        match reusable_filemap {
            Some(fm) => {
                cstore::ImportedFileMap {
                    original_start_pos: filemap_to_import.start_pos,
                    original_end_pos: filemap_to_import.end_pos,
                    translated_filemap: fm
                }
            }
            None => {
                // We can't reuse an existing FileMap, so allocate a new one
                // containing the information we need.
                let codemap::FileMap {
                    name,
                    start_pos,
                    end_pos,
                    lines,
                    multibyte_chars,
                    ..
                } = filemap_to_import;

                let source_length = (end_pos - start_pos).to_usize();

                // Translate line-start positions and multibyte character
                // position into frame of reference local to file.
                // `CodeMap::new_imported_filemap()` will then translate those
                // coordinates to their new global frame of reference when the
                // offset of the FileMap is known.
                let mut lines = lines.into_inner();
                for pos in &mut lines {
                    *pos = *pos - start_pos;
                }
                let mut multibyte_chars = multibyte_chars.into_inner();
                for mbc in &mut multibyte_chars {
                    mbc.pos = mbc.pos - start_pos;
                }

                let local_version = local_codemap.new_imported_filemap(name,
                                                                       source_length,
                                                                       lines,
                                                                       multibyte_chars);
                cstore::ImportedFileMap {
                    original_start_pos: start_pos,
                    original_end_pos: end_pos,
                    translated_filemap: local_version
                }
            }
        }
    }).collect();

    return imported_filemaps;

    fn are_equal_modulo_startpos(fm1: &codemap::FileMap,
                                 fm2: &codemap::FileMap)
                                 -> bool {
        if fm1.name != fm2.name {
            return false;
        }

        let lines1 = fm1.lines.borrow();
        let lines2 = fm2.lines.borrow();

        if lines1.len() != lines2.len() {
            return false;
        }

        for (&line1, &line2) in lines1.iter().zip(lines2.iter()) {
            if (line1 - fm1.start_pos) != (line2 - fm2.start_pos) {
                return false;
            }
        }

        let multibytes1 = fm1.multibyte_chars.borrow();
        let multibytes2 = fm2.multibyte_chars.borrow();

        if multibytes1.len() != multibytes2.len() {
            return false;
        }

        for (mb1, mb2) in multibytes1.iter().zip(multibytes2.iter()) {
            if (mb1.bytes != mb2.bytes) ||
               ((mb1.pos - fm1.start_pos) != (mb2.pos - fm2.start_pos)) {
                return false;
            }
        }

        true
    }
}
