// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use metadata::cstore;
use metadata::cstore::{CStore, CrateSource, MetadataBlob};
use metadata::decoder;
use metadata::loader;
use metadata::loader::CratePaths;

use std::rc::Rc;
use syntax::ast;
use syntax::abi;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{COMMAND_LINE_SP, Span, mk_sp};
use syntax::parse;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::visit;
use util::fs;
use log;

pub struct CrateReader<'a> {
    sess: &'a Session,
    next_crate_num: ast::CrateNum,
}

impl<'a, 'v> visit::Visitor<'v> for CrateReader<'a> {
    fn visit_view_item(&mut self, a: &ast::ViewItem) {
        self.process_view_item(a);
        visit::walk_view_item(self, a);
    }
    fn visit_item(&mut self, a: &ast::Item) {
        self.process_item(a);
        visit::walk_item(self, a);
    }
}

fn dump_crates(cstore: &CStore) {
    debug!("resolved crates:");
    cstore.iter_crate_data_origins(|_, data, opt_source| {
        debug!("  name: {}", data.name());
        debug!("  cnum: {}", data.cnum);
        debug!("  hash: {}", data.hash());
        opt_source.map(|cs| {
            let CrateSource { dylib, rlib, cnum: _ } = cs;
            dylib.map(|dl| debug!("  dylib: {}", dl.display()));
            rlib.map(|rl|  debug!("   rlib: {}", rl.display()));
        });
    })
}

fn should_link(i: &ast::ViewItem) -> bool {
    !attr::contains_name(&i.attrs[], "no_link")

}

struct CrateInfo {
    ident: String,
    name: String,
    id: ast::NodeId,
    should_link: bool,
}

pub fn validate_crate_name(sess: Option<&Session>, s: &str, sp: Option<Span>) {
    let err = |&: s: &str| {
        match (sp, sess) {
            (_, None) => panic!("{}", s),
            (Some(sp), Some(sess)) => sess.span_err(sp, s),
            (None, Some(sess)) => sess.err(s),
        }
    };
    if s.len() == 0 {
        err("crate name must not be empty");
    }
    for c in s.chars() {
        if c.is_alphanumeric() { continue }
        if c == '_' || c == '-' { continue }
        err(&format!("invalid character `{}` in crate name: `{}`", c, s)[]);
    }
    match sess {
        Some(sess) => sess.abort_if_errors(),
        None => {}
    }
}


fn register_native_lib(sess: &Session,
                       span: Option<Span>,
                       name: String,
                       kind: cstore::NativeLibraryKind) {
    if name.is_empty() {
        match span {
            Some(span) => {
                sess.span_err(span, "#[link(name = \"\")] given with \
                                     empty name");
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
            Some(span) => sess.span_err(span, msg),
            None => sess.err(msg),
        }
    }
    sess.cstore.add_used_library(name, kind);
}

pub struct PluginMetadata<'a> {
    sess: &'a Session,
    metadata: PMDSource,
    dylib: Option<Path>,
    info: CrateInfo,
    vi_span: Span,
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
    pub fn new(sess: &'a Session) -> CrateReader<'a> {
        CrateReader {
            sess: sess,
            next_crate_num: sess.cstore.next_crate_num(),
        }
    }

    // Traverses an AST, reading all the information about use'd crates and extern
    // libraries necessary for later resolving, typechecking, linking, etc.
    pub fn read_crates(&mut self, krate: &ast::Crate) {
        self.process_crate(krate);
        visit::walk_crate(self, krate);

        if log_enabled!(log::DEBUG) {
            dump_crates(&self.sess.cstore);
        }

        for &(ref name, kind) in self.sess.opts.libs.iter() {
            register_native_lib(self.sess, None, name.clone(), kind);
        }
    }

    fn process_crate(&self, c: &ast::Crate) {
        for a in c.attrs.iter().filter(|m| m.name() == "link_args") {
            match a.value_str() {
                Some(ref linkarg) => self.sess.cstore.add_used_link_args(linkarg.get()),
                None => { /* fallthrough */ }
            }
        }
    }

    fn process_view_item(&mut self, i: &ast::ViewItem) {
        if !should_link(i) {
            return;
        }

        match self.extract_crate_info(i) {
            Some(info) => {
                let (cnum, _, _) = self.resolve_crate(&None,
                                                      &info.ident[],
                                                      &info.name[],
                                                      None,
                                                      i.span,
                                                      PathKind::Crate);
                self.sess.cstore.add_extern_mod_stmt_cnum(info.id, cnum);
            }
            None => ()
        }
    }

    fn extract_crate_info(&self, i: &ast::ViewItem) -> Option<CrateInfo> {
        match i.node {
            ast::ViewItemExternCrate(ident, ref path_opt, id) => {
                let ident = token::get_ident(ident);
                debug!("resolving extern crate stmt. ident: {} path_opt: {:?}",
                       ident, path_opt);
                let name = match *path_opt {
                    Some((ref path_str, _)) => {
                        let name = path_str.get().to_string();
                        validate_crate_name(Some(self.sess), &name[],
                                            Some(i.span));
                        name
                    }
                    None => ident.get().to_string(),
                };
                Some(CrateInfo {
                    ident: ident.get().to_string(),
                    name: name,
                    id: id,
                    should_link: should_link(i),
                })
            }
            _ => None
        }
    }

    fn process_item(&self, i: &ast::Item) {
        match i.node {
            ast::ItemForeignMod(ref fm) => {
                if fm.abi == abi::Rust || fm.abi == abi::RustIntrinsic {
                    return;
                }

                // First, add all of the custom link_args attributes
                let link_args = i.attrs.iter()
                    .filter_map(|at| if at.name() == "link_args" {
                        Some(at)
                    } else {
                        None
                    })
                    .collect::<Vec<&ast::Attribute>>();
                for m in link_args.iter() {
                    match m.value_str() {
                        Some(linkarg) => self.sess.cstore.add_used_link_args(linkarg.get()),
                        None => { /* fallthrough */ }
                    }
                }

                // Next, process all of the #[link(..)]-style arguments
                let link_args = i.attrs.iter()
                    .filter_map(|at| if at.name() == "link" {
                        Some(at)
                    } else {
                        None
                    })
                    .collect::<Vec<&ast::Attribute>>();
                for m in link_args.iter() {
                    match m.meta_item_list() {
                        Some(items) => {
                            let kind = items.iter().find(|k| {
                                k.name() == "kind"
                            }).and_then(|a| a.value_str());
                            let kind = match kind {
                                Some(k) => {
                                    if k == "static" {
                                        cstore::NativeStatic
                                    } else if self.sess.target.target.options.is_like_osx
                                              && k == "framework" {
                                        cstore::NativeFramework
                                    } else if k == "framework" {
                                        cstore::NativeFramework
                                    } else if k == "dylib" {
                                        cstore::NativeUnknown
                                    } else {
                                        self.sess.span_err(m.span,
                                            &format!("unknown kind: `{}`",
                                                    k)[]);
                                        cstore::NativeUnknown
                                    }
                                }
                                None => cstore::NativeUnknown
                            };
                            let n = items.iter().find(|n| {
                                n.name() == "name"
                            }).and_then(|a| a.value_str());
                            let n = match n {
                                Some(n) => n,
                                None => {
                                    self.sess.span_err(m.span,
                                        "#[link(...)] specified without \
                                         `name = \"foo\"`");
                                    InternedString::new("foo")
                                }
                            };
                            register_native_lib(self.sess, Some(m.span),
                                                n.get().to_string(), kind);
                        }
                        None => {}
                    }
                }
            }
            _ => { }
        }
    }

    fn existing_match(&self, name: &str,
                      hash: Option<&Svh>) -> Option<ast::CrateNum> {
        let mut ret = None;
        self.sess.cstore.iter_crate_data(|cnum, data| {
            if data.name != name { return }

            match hash {
                Some(hash) if *hash == data.hash() => { ret = Some(cnum); return }
                Some(..) => return,
                None => {}
            }

            // When the hash is None we're dealing with a top-level dependency in
            // which case we may have a specification on the command line for this
            // library. Even though an upstream library may have loaded something of
            // the same name, we have to make sure it was loaded from the exact same
            // location as well.
            //
            // We're also sure to compare *paths*, not actual byte slices. The
            // `source` stores paths which are normalized which may be different
            // from the strings on the command line.
            let source = self.sess.cstore.get_used_crate_source(cnum).unwrap();
            match self.sess.opts.externs.get(name) {
                Some(locs) => {
                    let found = locs.iter().any(|l| {
                        let l = fs::realpath(&Path::new(&l[])).ok();
                        l == source.dylib || l == source.rlib
                    });
                    if found {
                        ret = Some(cnum);
                    }
                }
                None => ret = Some(cnum),
            }
        });
        return ret;
    }

    fn register_crate(&mut self,
                      root: &Option<CratePaths>,
                      ident: &str,
                      name: &str,
                      span: Span,
                      lib: loader::Library)
                      -> (ast::CrateNum, Rc<cstore::crate_metadata>,
                          cstore::CrateSource) {
        // Claim this crate number and cache it
        let cnum = self.next_crate_num;
        self.next_crate_num += 1;

        // Stash paths for top-most crate locally if necessary.
        let crate_paths = if root.is_none() {
            Some(CratePaths {
                ident: ident.to_string(),
                dylib: lib.dylib.clone(),
                rlib:  lib.rlib.clone(),
            })
        } else {
            None
        };
        // Maintain a reference to the top most crate.
        let root = if root.is_some() { root } else { &crate_paths };

        let cnum_map = self.resolve_crate_deps(root, lib.metadata.as_slice(), span);

        let loader::Library{ dylib, rlib, metadata } = lib;

        let cmeta = Rc::new( cstore::crate_metadata {
            name: name.to_string(),
            data: metadata,
            cnum_map: cnum_map,
            cnum: cnum,
            span: span,
        });

        let source = cstore::CrateSource {
            dylib: dylib,
            rlib: rlib,
            cnum: cnum,
        };

        self.sess.cstore.set_crate_data(cnum, cmeta.clone());
        self.sess.cstore.add_used_crate_source(source.clone());
        (cnum, cmeta, source)
    }

    fn resolve_crate(&mut self,
                     root: &Option<CratePaths>,
                     ident: &str,
                     name: &str,
                     hash: Option<&Svh>,
                     span: Span,
                     kind: PathKind)
                         -> (ast::CrateNum, Rc<cstore::crate_metadata>,
                             cstore::CrateSource) {
        match self.existing_match(name, hash) {
            None => {
                let mut load_ctxt = loader::Context {
                    sess: self.sess,
                    span: span,
                    ident: ident,
                    crate_name: name,
                    hash: hash.map(|a| &*a),
                    filesearch: self.sess.target_filesearch(kind),
                    target: &self.sess.target.target,
                    triple: &self.sess.opts.target_triple[],
                    root: root,
                    rejected_via_hash: vec!(),
                    rejected_via_triple: vec!(),
                    should_match_name: true,
                };
                let library = load_ctxt.load_library_crate();
                self.register_crate(root, ident, name, span, library)
            }
            Some(cnum) => (cnum,
                           self.sess.cstore.get_crate_data(cnum),
                           self.sess.cstore.get_used_crate_source(cnum).unwrap())
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
                                                   &dep.name[],
                                                   &dep.name[],
                                                   Some(&dep.hash),
                                                   span,
                                                   PathKind::Dependency);
            (dep.cnum, local_cnum)
        }).collect()
    }

    pub fn read_plugin_metadata<'b>(&'b mut self,
                                    krate: CrateOrString<'b>) -> PluginMetadata<'b> {
        let (info, span) = match krate {
            CrateOrString::Krate(c) => {
                (self.extract_crate_info(c).unwrap(), c.span)
            }
            CrateOrString::Str(s) => {
                (CrateInfo {
                     name: s.to_string(),
                     ident: s.to_string(),
                     id: ast::DUMMY_NODE_ID,
                     should_link: true,
                 }, COMMAND_LINE_SP)
            }
        };
        let target_triple = &self.sess.opts.target_triple[];
        let is_cross = target_triple != config::host_triple();
        let mut should_link = info.should_link && !is_cross;
        let mut target_only = false;
        let ident = info.ident.clone();
        let name = info.name.clone();
        let mut load_ctxt = loader::Context {
            sess: self.sess,
            span: span,
            ident: &ident[],
            crate_name: &name[],
            hash: None,
            filesearch: self.sess.host_filesearch(PathKind::Crate),
            target: &self.sess.host,
            triple: config::host_triple(),
            root: &None,
            rejected_via_hash: vec!(),
            rejected_via_triple: vec!(),
            should_match_name: true,
        };
        let library = match load_ctxt.maybe_load_library_crate() {
            Some(l) => l,
            None if is_cross => {
                // Try loading from target crates. This will abort later if we try to
                // load a plugin registrar function,
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
        let register = should_link && self.existing_match(info.name.as_slice(), None).is_none();
        let metadata = if register {
            // Register crate now to avoid double-reading metadata
            let (_, cmd, _) = self.register_crate(&None, &info.ident[],
                                &info.name[], span, library);
            PMDSource::Registered(cmd)
        } else {
            // Not registering the crate; just hold on to the metadata
            PMDSource::Owned(library.metadata)
        };

        PluginMetadata {
            sess: self.sess,
            metadata: metadata,
            dylib: dylib,
            info: info,
            vi_span: span,
            target_only: target_only,
        }
    }
}

#[derive(Copy)]
pub enum CrateOrString<'a> {
    Krate(&'a ast::ViewItem),
    Str(&'a str)
}

impl<'a> PluginMetadata<'a> {
    /// Read exported macros
    pub fn exported_macros(&self) -> Vec<ast::MacroDef> {
        let imported_from = Some(token::intern(&self.info.ident[]).ident());
        let source_name = format!("<{} macros>", &self.info.ident[]);
        let mut macros = vec![];
        decoder::each_exported_macro(self.metadata.as_slice(),
                                     &*self.sess.cstore.intr,
            |name, attrs, body| {
                // NB: Don't use parse::parse_tts_from_source_str because it parses with
                // quote_depth > 0.
                let mut p = parse::new_parser_from_source_str(&self.sess.parse_sess,
                                                              self.sess.opts.cfg.clone(),
                                                              source_name.clone(),
                                                              body);
                let lo = p.span.lo;
                let body = p.parse_all_token_trees();
                let span = mk_sp(lo, p.last_span.hi);
                p.abort_if_errors();
                macros.push(ast::MacroDef {
                    ident: name.ident(),
                    attrs: attrs,
                    id: ast::DUMMY_NODE_ID,
                    span: span,
                    imported_from: imported_from,
                    // overridden in plugin/load.rs
                    export: false,
                    use_locally: false,

                    body: body,
                });
                true
            }
        );
        macros
    }

    /// Look for a plugin registrar. Returns library path and symbol name.
    pub fn plugin_registrar(&self) -> Option<(Path, String)> {
        if self.target_only {
            // Need to abort before syntax expansion.
            let message = format!("plugin crate `{}` is not available for triple `{}` \
                                   (only found {})",
                                  self.info.ident,
                                  config::host_triple(),
                                  self.sess.opts.target_triple);
            self.sess.span_err(self.vi_span, &message[]);
            self.sess.abort_if_errors();
        }

        let registrar = decoder::get_plugin_registrar_fn(self.metadata.as_slice())
            .map(|id| decoder::get_symbol(self.metadata.as_slice(), id));

        match (self.dylib.as_ref(), registrar) {
            (Some(dylib), Some(reg)) => Some((dylib.clone(), reg)),
            (None, Some(_)) => {
                let message = format!("plugin crate `{}` only found in rlib format, \
                                       but must be available in dylib format",
                                       self.info.ident);
                self.sess.span_err(self.vi_span, &message[]);
                // No need to abort because the loading code will just ignore this
                // empty dylib.
                None
            }
            _ => None,
        }
    }
}
