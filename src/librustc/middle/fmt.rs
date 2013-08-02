// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module implements the collection of #[fmt]-tagged traits in both the
//! local crate and external crates. This merges these two lists of fmt-traits
//! into one `TraitMap` which is then used during resolve and finally shoved
//! into the tcx.

use driver::session::Session;
use metadata::csearch;
use metadata::cstore::iter_crate_data;

use syntax::ast;
use syntax::ast_util;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::span;
use syntax::visit;
use syntax::parse::token::str_to_ident;

use std::hashmap::HashMap;
use std::util;

/// Mapping of format string names to a (def_id, def) pair. The def_id is the
/// actual id of the formatting function while the `def` is the definition
/// corresponding for it.
pub type TraitMap = HashMap<@str, (ast::def_id, ast::def)>;

struct Context {
    /// Cached ident for the word "fmt" (what the format function is called)
    fmt_ident: ast::ident,
    sess: Session,

    /// The `fmts` map is returned at the end of the collection, and in the
    /// interim the `fmt_loc` map keeps track of where the traits came from so
    /// collisions can be diagnosed sanely.
    fmts: TraitMap,
    fmt_loc: HashMap<@str, Either<span, @str>>,
}

impl Context {
    /// Registers a new format-trait
    fn register(&mut self, name: @str, src: Either<span, @str>,
                fmt_id: ast::def_id, trait_id: ast::def_id) {
        match self.fmts.find(&name) {
            Some(*) => {
                match src {
                    Left(sp) => {
                        self.sess.span_err(sp, fmt!("duplicate fmt trait \
                                                     for `%s`", name));
                    }
                    Right(src) => {
                        self.sess.err(fmt!("duplicate fmt trait `%s` found in \
                                            external crate `%s`", name, src));
                    }
                }
                match *self.fmt_loc.get(&name) {
                    Left(sp) => {
                        self.sess.span_note(sp, "previous definition here");
                    }
                    Right(name) => {
                        self.sess.note(fmt!("previous definition found in the \
                                             external crate `%s`", name));
                    }
                }
                return;
            }
            None => {}
        }
        let def = ast::def_static_method(fmt_id, Some(trait_id),
                                         ast::impure_fn);
        debug!("found %s at %?", name, src);
        self.fmts.insert(name, (fmt_id, def));
        self.fmt_loc.insert(name, src);
    }

    /// Iterates over the local crate to identify all #[fmt]-tagged traits and
    /// adds them to the local trait map
    fn collect_local_traits(@mut self, crate: &ast::Crate) {
        visit::visit_crate(crate, (self, visit::mk_vt(@visit::Visitor {
            visit_item: |it, (cx, vt)| {
                let cx: @mut Context = cx;
                for it.attrs.iter().advance |attr| {
                    if "fmt" != attr.name() { loop }
                    match attr.value_str() {
                        Some(name) => {
                            match it.node {
                                ast::item_trait(_, _, ref methods) => {
                                    cx.find_fmt_method(name, it.span, it.id,
                                                       *methods);
                                }
                                _ => {
                                    cx.sess.span_err(attr.span,
                                                     "fmt attribute can only be \
                                                      specified on traits");
                                }
                            }
                        }
                        None => {
                            cx.sess.span_err(attr.span,
                                             "fmt attribute must have a value \
                                              specified (fmt=\"...\")");
                        }
                    }
                }
                visit::visit_item(it, (cx, vt));
            },
            .. *visit::default_visitor()
        })));
    }

    fn find_fmt_method(@mut self, name: @str, sp: span, id: ast::NodeId,
                       methods: &[ast::trait_method]) {
        fn check_purity(sess: Session, p: ast::purity, sp: span) {
            match p {
                ast::unsafe_fn => {
                    sess.span_err(sp, "the `fmt` function must not be unsafe");
                }
                ast::extern_fn => {
                    sess.span_err(sp, "the `fmt` function must not be extern");
                }
                ast::impure_fn => {}
            }
        }

        let mut found = false;
        for methods.iter().advance |m| {
            match *m {
                ast::required(ref m) if m.ident == self.fmt_ident => {
                    check_purity(self.sess, m.purity, sp);
                    self.register(name, Left(sp), ast_util::local_def(m.id),
                                  ast_util::local_def(id));
                    found = true;
                }
                ast::provided(ref m) if m.ident == self.fmt_ident => {
                    check_purity(self.sess, m.purity, sp);
                    self.register(name, Left(sp), ast_util::local_def(m.id),
                                  ast_util::local_def(id));
                    found = true;
                }

                ast::provided(*) | ast::required(*) => {}
            }
        }
        if !found {
            self.sess.span_err(sp, "no function named `fmt` on format trait");
        }
    }

    /// Iterates over all of the external crates and loads all of the external
    /// fmt traits into the local maps.
    fn collect_external_traits(&mut self) {
        let cstore = self.sess.cstore;
        do iter_crate_data(cstore) |n, metadata| {
            for csearch::each_fmt_trait(cstore, n) |name, fmt_id, trait_id| {
                let fmt_id = ast::def_id { crate: n, node: fmt_id };
                let trait_id = ast::def_id { crate: n, node: trait_id };
                self.register(name, Right(metadata.name), fmt_id, trait_id);
            }
        }
    }
}

pub fn find_fmt_traits(crate: &ast::Crate, session: Session) -> @TraitMap {
    let cx = @mut Context {
        fmt_ident: str_to_ident("fmt"),
        sess: session,
        fmts: HashMap::new(),
        fmt_loc: HashMap::new(),
    };

    cx.collect_external_traits();
    cx.collect_local_traits(crate);
    session.abort_if_errors();

    @util::replace(&mut cx.fmts, HashMap::new())
}
