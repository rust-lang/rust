// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use attr;
use codemap::DUMMY_SP;
use codemap;
use fold::Folder;
use fold;
use parse::token::InternedString;
use parse::token::special_idents;
use parse::token;
use ptr::P;
use util::small_vector::SmallVector;

pub fn maybe_inject_crates_ref(krate: ast::Crate, alt_std_name: Option<String>)
                               -> ast::Crate {
    if use_std(&krate) {
        inject_crates_ref(krate, alt_std_name)
    } else {
        krate
    }
}

pub fn maybe_inject_prelude(krate: ast::Crate) -> ast::Crate {
    if use_std(&krate) {
        inject_prelude(krate)
    } else {
        krate
    }
}

pub fn use_std(krate: &ast::Crate) -> bool {
    !attr::contains_name(&krate.attrs[], "no_std")
}

fn no_prelude(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "no_implicit_prelude")
}

struct StandardLibraryInjector {
    alt_std_name: Option<String>,
}

impl fold::Folder for StandardLibraryInjector {
    fn fold_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {

        // The name to use in `extern crate "name" as std;`
        let actual_crate_name = match self.alt_std_name {
            Some(ref s) => token::intern_and_get_ident(&s[..]),
            None => token::intern_and_get_ident("std"),
        };

        krate.module.items.insert(0, P(ast::Item {
            id: ast::DUMMY_NODE_ID,
            ident: token::str_to_ident("std"),
            attrs: vec!(
                attr::mk_attr_outer(attr::mk_attr_id(), attr::mk_word_item(
                        InternedString::new("macro_use")))),
            node: ast::ItemExternCrate(Some((actual_crate_name, ast::CookedStr))),
            vis: ast::Inherited,
            span: DUMMY_SP
        }));

        krate
    }
}

fn inject_crates_ref(krate: ast::Crate, alt_std_name: Option<String>) -> ast::Crate {
    let mut fold = StandardLibraryInjector {
        alt_std_name: alt_std_name
    };
    fold.fold_crate(krate)
}

struct PreludeInjector;


impl fold::Folder for PreludeInjector {
    fn fold_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        // only add `use std::prelude::*;` if there wasn't a
        // `#![no_implicit_prelude]` at the crate level.
        // fold_mod() will insert glob path.
        if !no_prelude(&krate.attrs[]) {
            krate.module = self.fold_mod(krate.module);
        }
        krate
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        if !no_prelude(&item.attrs[]) {
            // only recur if there wasn't `#![no_implicit_prelude]`
            // on this item, i.e. this means that the prelude is not
            // implicitly imported though the whole subtree
            fold::noop_fold_item(item, self)
        } else {
            SmallVector::one(item)
        }
    }

    fn fold_mod(&mut self, mut mod_: ast::Mod) -> ast::Mod {
        let prelude_path = ast::Path {
            span: DUMMY_SP,
            global: false,
            segments: vec![
                ast::PathSegment {
                    identifier: token::str_to_ident("std"),
                    parameters: ast::PathParameters::none(),
                },
                ast::PathSegment {
                    identifier: token::str_to_ident("prelude"),
                    parameters: ast::PathParameters::none(),
                },
                ast::PathSegment {
                    identifier: token::str_to_ident("v1"),
                    parameters: ast::PathParameters::none(),
                },
            ],
        };

        let vp = P(codemap::dummy_spanned(ast::ViewPathGlob(prelude_path)));
        mod_.items.insert(0, P(ast::Item {
            id: ast::DUMMY_NODE_ID,
            ident: special_idents::invalid,
            node: ast::ItemUse(vp),
            attrs: vec![ast::Attribute {
                span: DUMMY_SP,
                node: ast::Attribute_ {
                    id: attr::mk_attr_id(),
                    style: ast::AttrOuter,
                    value: P(ast::MetaItem {
                        span: DUMMY_SP,
                        node: ast::MetaWord(token::get_name(
                                special_idents::prelude_import.name)),
                    }),
                    is_sugared_doc: false,
                },
            }],
            vis: ast::Inherited,
            span: DUMMY_SP,
        }));

        fold::noop_fold_mod(mod_, self)
    }
}

fn inject_prelude(krate: ast::Crate) -> ast::Crate {
    let mut fold = PreludeInjector;
    fold.fold_crate(krate)
}
