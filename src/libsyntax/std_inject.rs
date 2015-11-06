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
use codemap::{DUMMY_SP, Span, ExpnInfo, NameAndSpan, MacroAttribute};
use codemap;
use fold::Folder;
use fold;
use parse::token::{intern, InternedString, special_idents};
use parse::{token, ParseSess};
use ptr::P;
use util::small_vector::SmallVector;

/// Craft a span that will be ignored by the stability lint's
/// call to codemap's is_internal check.
/// The expanded code uses the unstable `#[prelude_import]` attribute.
fn ignored_span(sess: &ParseSess, sp: Span) -> Span {
    let info = ExpnInfo {
        call_site: DUMMY_SP,
        callee: NameAndSpan {
            format: MacroAttribute(intern("std_inject")),
            span: None,
            allow_internal_unstable: true,
        }
    };
    let expn_id = sess.codemap().record_expansion(info);
    let mut sp = sp;
    sp.expn_id = expn_id;
    return sp;
}

pub fn maybe_inject_crates_ref(krate: ast::Crate, alt_std_name: Option<String>)
                               -> ast::Crate {
    if no_core(&krate) {
        krate
    } else {
        let name = if no_std(&krate) {"core"} else {"std"};
        let mut fold = CrateInjector {
            item_name: token::str_to_ident(name),
            crate_name: token::intern(&alt_std_name.unwrap_or(String::from(name))),
        };
        fold.fold_crate(krate)
    }
}

pub fn maybe_inject_prelude(sess: &ParseSess, krate: ast::Crate) -> ast::Crate {
    if no_core(&krate) {
        krate
    } else {
        let name = if no_std(&krate) {"core"} else {"std"};
        let mut fold = PreludeInjector {
            span: ignored_span(sess, DUMMY_SP),
            crate_identifier: token::str_to_ident(name),
        };
        fold.fold_crate(krate)
    }
}

pub fn no_core(krate: &ast::Crate) -> bool {
    attr::contains_name(&krate.attrs, "no_core")
}

pub fn no_std(krate: &ast::Crate) -> bool {
    attr::contains_name(&krate.attrs, "no_std") || no_core(krate)
}

fn no_prelude(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "no_implicit_prelude")
}

struct CrateInjector {
    item_name: ast::Ident,
    crate_name: ast::Name,
}

impl fold::Folder for CrateInjector {
    fn fold_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        krate.module.items.insert(0, P(ast::Item {
            id: ast::DUMMY_NODE_ID,
            ident: self.item_name,
            attrs: vec!(
                attr::mk_attr_outer(attr::mk_attr_id(), attr::mk_word_item(
                        InternedString::new("macro_use")))),
            node: ast::ItemExternCrate(Some(self.crate_name)),
            vis: ast::Inherited,
            span: DUMMY_SP
        }));

        krate
    }
}

struct PreludeInjector {
    span: Span,
    crate_identifier: ast::Ident,
}

impl fold::Folder for PreludeInjector {
    fn fold_crate(&mut self, mut krate: ast::Crate) -> ast::Crate {
        // only add `use std::prelude::*;` if there wasn't a
        // `#![no_implicit_prelude]` at the crate level.
        // fold_mod() will insert glob path.
        if !no_prelude(&krate.attrs) {
            krate.module = self.fold_mod(krate.module);
        }
        krate
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        if !no_prelude(&item.attrs) {
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
            span: self.span,
            global: false,
            segments: vec![
                ast::PathSegment {
                    identifier: self.crate_identifier,
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
                span: self.span,
                node: ast::Attribute_ {
                    id: attr::mk_attr_id(),
                    style: ast::AttrStyle::Outer,
                    value: P(ast::MetaItem {
                        span: self.span,
                        node: ast::MetaWord(special_idents::prelude_import.name.as_str()),
                    }),
                    is_sugared_doc: false,
                },
            }],
            vis: ast::Inherited,
            span: self.span,
        }));

        fold::noop_fold_mod(mod_, self)
    }
}
