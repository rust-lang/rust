// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::Session;

use std::vec_ng::Vec;
use std::vec_ng;
use syntax::ast;
use syntax::attr;
use syntax::codemap::DUMMY_SP;
use syntax::codemap;
use syntax::fold::Folder;
use syntax::fold;
use syntax::opt_vec;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::util::small_vector::SmallVector;

pub static VERSION: &'static str = "0.10-pre";

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

fn use_uv(krate: &ast::Crate) -> bool {
    !attr::contains_name(krate.attrs.as_slice(), "no_uv")
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
                                                      VERSION)),
                  ast::CookedStr))
        }
    }
}

impl<'a> fold::Folder for StandardLibraryInjector<'a> {
    fn fold_crate(&mut self, krate: ast::Crate) -> ast::Crate {
        let mut vis = vec!(ast::ViewItem {
            node: ast::ViewItemExternCrate(token::str_to_ident("std"),
                                         with_version("std"),
                                         ast::DUMMY_NODE_ID),
            attrs: vec!(
                attr::mk_attr(attr::mk_list_item(
                        InternedString::new("phase"),
                        vec!(
                            attr::mk_word_item(InternedString::new("syntax")),
                            attr::mk_word_item(InternedString::new("link")
                        ))))),
            vis: ast::Inherited,
            span: DUMMY_SP
        });

        if use_uv(&krate) && !self.sess.building_library.get() {
            vis.push(ast::ViewItem {
                node: ast::ViewItemExternCrate(token::str_to_ident("green"),
                                             with_version("green"),
                                             ast::DUMMY_NODE_ID),
                attrs: Vec::new(),
                vis: ast::Inherited,
                span: DUMMY_SP
            });
            vis.push(ast::ViewItem {
                node: ast::ViewItemExternCrate(token::str_to_ident("rustuv"),
                                             with_version("rustuv"),
                                             ast::DUMMY_NODE_ID),
                attrs: Vec::new(),
                vis: ast::Inherited,
                span: DUMMY_SP
            });
        }

        vis.push_all_move(krate.module.view_items.clone());
        let new_module = ast::Mod {
            view_items: vis,
            ..krate.module.clone()
        };

        ast::Crate {
            module: new_module,
            ..krate
        }
    }
}

fn inject_crates_ref(sess: &Session, krate: ast::Crate) -> ast::Crate {
    let mut fold = StandardLibraryInjector {
        sess: sess,
    };
    fold.fold_crate(krate)
}

struct PreludeInjector<'a> {
    sess: &'a Session,
}


impl<'a> fold::Folder for PreludeInjector<'a> {
    fn fold_crate(&mut self, krate: ast::Crate) -> ast::Crate {
        if !no_prelude(krate.attrs.as_slice()) {
            // only add `use std::prelude::*;` if there wasn't a
            // `#[no_implicit_prelude];` at the crate level.
            ast::Crate {
                module: self.fold_mod(&krate.module),
                ..krate
            }
        } else {
            krate
        }
    }

    fn fold_item(&mut self, item: @ast::Item) -> SmallVector<@ast::Item> {
        if !no_prelude(item.attrs.as_slice()) {
            // only recur if there wasn't `#[no_implicit_prelude];`
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
                    types: opt_vec::Empty,
                },
                ast::PathSegment {
                    identifier: token::str_to_ident("prelude"),
                    lifetimes: Vec::new(),
                    types: opt_vec::Empty,
                }),
        };

        let vp = @codemap::dummy_spanned(ast::ViewPathGlob(prelude_path, ast::DUMMY_NODE_ID));
        let vi2 = ast::ViewItem {
            node: ast::ViewItemUse(vec!(vp)),
            attrs: Vec::new(),
            vis: ast::Inherited,
            span: DUMMY_SP,
        };

        let vis = vec_ng::append(vec!(vi2), module.view_items.as_slice());

        // FIXME #2543: Bad copy.
        let new_module = ast::Mod {
            view_items: vis,
            ..(*module).clone()
        };
        fold::noop_fold_mod(&new_module, self)
    }
}

fn inject_prelude(sess: &Session, krate: ast::Crate) -> ast::Crate {
    let mut fold = PreludeInjector {
        sess: sess,
    };
    fold.fold_crate(krate)
}
