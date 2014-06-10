// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::config;
use driver::session::Session;

use syntax::ast;
use syntax::attr;
use syntax::codemap::DUMMY_SP;
use syntax::codemap;
use syntax::fold::Folder;
use syntax::fold;
use syntax::owned_slice::OwnedSlice;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::util::small_vector::SmallVector;

use std::mem;

pub static VERSION: &'static str = "0.11.0-pre";

pub fn maybe_inject_crates_ref(sess: &Session, krate: ast::Crate)
                               -> ast::Crate {
    if use_std(&krate) {
        inject_crates_ref(sess, krate)
    } else {
        krate
    }
}

pub fn maybe_inject_prelude(sess: &Session, krate: ast::Crate) -> ast::Crate {
    if use_std(&krate) {
        inject_prelude(sess, krate)
    } else {
        krate
    }
}

fn use_std(krate: &ast::Crate) -> bool {
    !attr::contains_name(krate.attrs.as_slice(), "no_std")
}

fn use_start(krate: &ast::Crate) -> bool {
    !attr::contains_name(krate.attrs.as_slice(), "no_start")
}

fn no_prelude(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "no_implicit_prelude")
}

struct StandardLibraryInjector<'a> {
    sess: &'a Session,
}

pub fn with_version(krate: &str) -> Option<(InternedString, ast::StrStyle)> {
    match option_env!("CFG_DISABLE_INJECT_STD_VERSION") {
        Some("1") => None,
        _ => {
            Some((token::intern_and_get_ident(format!("{}\\#{}",
                                                      krate,
                                                      VERSION).as_slice()),
                  ast::CookedStr))
        }
    }
}

impl<'a> fold::Folder for StandardLibraryInjector<'a> {
    fn fold_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        let mut vis = vec!(ast::ViewItem {
            node: ast::ViewItemExternCrate(token::str_to_ident("std"),
                                         with_version("std"),
                                         ast::DUMMY_NODE_ID),
            attrs: vec!(
                attr::mk_attr_outer(attr::mk_attr_id(), attr::mk_list_item(
                        InternedString::new("phase"),
                        vec!(
                            attr::mk_word_item(InternedString::new("plugin")),
                            attr::mk_word_item(InternedString::new("link")
                        ))))),
            vis: ast::Inherited,
            span: DUMMY_SP
        });

        let any_exe = self.sess.crate_types.borrow().iter().any(|ty| {
            *ty == config::CrateTypeExecutable
        });
        if use_start(&krate) && any_exe {
            vis.push(ast::ViewItem {
                node: ast::ViewItemExternCrate(token::str_to_ident("native"),
                                             with_version("native"),
                                             ast::DUMMY_NODE_ID),
                attrs: Vec::new(),
                vis: ast::Inherited,
                span: DUMMY_SP
            });
        }

        // `extern crate` must be precede `use` items
        mem::swap(&mut vis, &mut krate.module.view_items);
        krate.module.view_items.push_all_move(vis);

        // don't add #![no_std] here, that will block the prelude injection later.
        // Add it during the prelude injection instead.

        // Add #![feature(phase)] here, because we use #[phase] on extern crate std.
        let feat_phase_attr = attr::mk_attr_inner(attr::mk_attr_id(),
                                                  attr::mk_list_item(
                                  InternedString::new("feature"),
                                  vec![attr::mk_word_item(InternedString::new("phase"))],
                              ));
        // std_inject runs after feature checking so manually mark this attr
        attr::mark_used(&feat_phase_attr);
        krate.attrs.push(feat_phase_attr);

        krate
    }
}

fn inject_crates_ref(sess: &Session, krate: ast::Crate) -> ast::Crate {
    let mut fold = StandardLibraryInjector {
        sess: sess,
    };
    fold.fold_crate(krate)
}

struct PreludeInjector<'a>;


impl<'a> fold::Folder for PreludeInjector<'a> {
    fn fold_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        // Add #![no_std] here, so we don't re-inject when compiling pretty-printed source.
        // This must happen here and not in StandardLibraryInjector because this
        // fold happens second.

        let no_std_attr = attr::mk_attr_inner(attr::mk_attr_id(),
                                              attr::mk_word_item(InternedString::new("no_std")));
        // std_inject runs after feature checking so manually mark this attr
        attr::mark_used(&no_std_attr);
        krate.attrs.push(no_std_attr);

        if !no_prelude(krate.attrs.as_slice()) {
            // only add `use std::prelude::*;` if there wasn't a
            // `#![no_implicit_prelude]` at the crate level.

            // fold_mod() will insert glob path.
            let globs_attr = attr::mk_attr_inner(attr::mk_attr_id(),
                                                 attr::mk_list_item(
                InternedString::new("feature"),
                vec!(
                    attr::mk_word_item(InternedString::new("globs")),
                )));
            // std_inject runs after feature checking so manually mark this attr
            attr::mark_used(&globs_attr);
            krate.attrs.push(globs_attr);

            krate.module = self.fold_mod(&krate.module);
        }
        krate
    }

    fn fold_item(&mut self, item: @ast::Item) -> SmallVector<@ast::Item> {
        if !no_prelude(item.attrs.as_slice()) {
            // only recur if there wasn't `#![no_implicit_prelude]`
            // on this item, i.e. this means that the prelude is not
            // implicitly imported though the whole subtree
            fold::noop_fold_item(item, self)
        } else {
            SmallVector::one(item)
        }
    }

    fn fold_mod(&mut self, module: &ast::Mod) -> ast::Mod {
        let prelude_path = ast::Path {
            span: DUMMY_SP,
            global: false,
            segments: vec!(
                ast::PathSegment {
                    identifier: token::str_to_ident("std"),
                    lifetimes: Vec::new(),
                    types: OwnedSlice::empty(),
                },
                ast::PathSegment {
                    identifier: token::str_to_ident("prelude"),
                    lifetimes: Vec::new(),
                    types: OwnedSlice::empty(),
                }),
        };

        let vp = @codemap::dummy_spanned(ast::ViewPathGlob(prelude_path, ast::DUMMY_NODE_ID));
        let vi2 = ast::ViewItem {
            node: ast::ViewItemUse(vp),
            attrs: Vec::new(),
            vis: ast::Inherited,
            span: DUMMY_SP,
        };

        let (crates, uses) = module.view_items.partitioned(|x| {
            match x.node {
                ast::ViewItemExternCrate(..) => true,
                _ => false,
            }
        });

        // add vi2 after any `extern crate` but before any `use`
        let mut view_items = crates;
        view_items.push(vi2);
        view_items.push_all_move(uses);

        let new_module = ast::Mod {
            view_items: view_items,
            ..(*module).clone()
        };
        fold::noop_fold_mod(&new_module, self)
    }
}

fn inject_prelude(_: &Session, krate: ast::Crate) -> ast::Crate {
    let mut fold = PreludeInjector;
    fold.fold_crate(krate)
}
