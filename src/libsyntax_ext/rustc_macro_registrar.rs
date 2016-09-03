// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

use errors;
use syntax::ast::{self, Ident, NodeId};
use syntax::codemap::{ExpnInfo, NameAndSpan, MacroAttribute};
use syntax::ext::base::{ExtCtxt, DummyMacroLoader};
use syntax::ext::build::AstBuilder;
use syntax::ext::expand::ExpansionConfig;
use syntax::parse::ParseSess;
use syntax::parse::token::{self, InternedString};
use syntax::feature_gate::Features;
use syntax::ptr::P;
use syntax_pos::{Span, DUMMY_SP};
use syntax::visit::{self, Visitor};

use deriving;

struct CustomDerive {
    trait_name: InternedString,
    function_name: Ident,
    span: Span,
}

struct CollectCustomDerives<'a> {
    derives: Vec<CustomDerive>,
    in_root: bool,
    handler: &'a errors::Handler,
    is_rustc_macro_crate: bool,
}

pub fn modify(sess: &ParseSess,
              mut krate: ast::Crate,
              is_rustc_macro_crate: bool,
              num_crate_types: usize,
              handler: &errors::Handler,
              features: &Features) -> ast::Crate {
    let mut loader = DummyMacroLoader;
    let mut cx = ExtCtxt::new(sess,
                              Vec::new(),
                              ExpansionConfig::default("rustc_macro".to_string()),
                              &mut loader);

    let mut collect = CollectCustomDerives {
        derives: Vec::new(),
        in_root: true,
        handler: handler,
        is_rustc_macro_crate: is_rustc_macro_crate,
    };
    visit::walk_crate(&mut collect, &krate);

    if !is_rustc_macro_crate {
        return krate
    } else if !features.rustc_macro {
        let mut err = handler.struct_err("the `rustc-macro` crate type is \
                                          experimental");
        err.help("add #![feature(rustc_macro)] to the crate attributes to \
                  enable");
        err.emit();
    }

    if num_crate_types > 1 {
        handler.err("cannot mix `rustc-macro` crate type with others");
    }

    krate.module.items.push(mk_registrar(&mut cx, &collect.derives));

    if krate.exported_macros.len() > 0 {
        handler.err("cannot export macro_rules! macros from a `rustc-macro` \
                     crate type currently");
    }

    return krate
}

impl<'a> CollectCustomDerives<'a> {
    fn check_not_pub_in_root(&self, vis: &ast::Visibility, sp: Span) {
        if self.is_rustc_macro_crate &&
           self.in_root &&
           *vis == ast::Visibility::Public {
            self.handler.span_err(sp,
                                  "`rustc-macro` crate types cannot \
                                   export any items other than functions \
                                   tagged with `#[rustc_macro_derive]` \
                                   currently");
        }
    }
}

impl<'a> Visitor for CollectCustomDerives<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        // First up, make sure we're checking a bare function. If we're not then
        // we're just not interested in this item.
        //
        // If we find one, try to locate a `#[rustc_macro_derive]` attribute on
        // it.
        match item.node {
            ast::ItemKind::Fn(..) => {}
            _ => {
                self.check_not_pub_in_root(&item.vis, item.span);
                return visit::walk_item(self, item)
            }
        }

        let mut attrs = item.attrs.iter()
                            .filter(|a| a.check_name("rustc_macro_derive"));
        let attr = match attrs.next() {
            Some(attr) => attr,
            None => {
                self.check_not_pub_in_root(&item.vis, item.span);
                return visit::walk_item(self, item)
            }
        };

        if let Some(a) = attrs.next() {
            self.handler.span_err(a.span(), "multiple `#[rustc_macro_derive]` \
                                             attributes found");
        }

        if !self.is_rustc_macro_crate {
            self.handler.span_err(attr.span(),
                                  "the `#[rustc_macro_derive]` attribute is \
                                   only usable with crates of the `rustc-macro` \
                                   crate type");
        }

        // Once we've located the `#[rustc_macro_derive]` attribute, verify
        // that it's of the form `#[rustc_macro_derive(Foo)]`
        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => {
                self.handler.span_err(attr.span(),
                                      "attribute must be of form: \
                                       #[rustc_macro_derive(TraitName)]");
                return
            }
        };
        if list.len() != 1 {
            self.handler.span_err(attr.span(),
                                  "attribute must only have one argument");
            return
        }
        let attr = &list[0];
        let trait_name = match attr.name() {
            Some(name) => name,
            _ => {
                self.handler.span_err(attr.span(), "not a meta item");
                return
            }
        };
        if !attr.is_word() {
            self.handler.span_err(attr.span(), "must only be one word");
        }

        if deriving::is_builtin_trait(&trait_name) {
            self.handler.span_err(attr.span(),
                                  "cannot override a built-in #[derive] mode");
        }

        if self.derives.iter().any(|d| d.trait_name == trait_name) {
            self.handler.span_err(attr.span(),
                                  "derive mode defined twice in this crate");
        }

        if self.in_root {
            self.derives.push(CustomDerive {
                span: item.span,
                trait_name: trait_name,
                function_name: item.ident,
            });
        } else {
            let msg = "functions tagged with `#[rustc_macro_derive]` must \
                       currently reside in the root of the crate";
            self.handler.span_err(item.span, msg);
        }

        visit::walk_item(self, item);
    }

    fn visit_mod(&mut self, m: &ast::Mod, _s: Span, id: NodeId) {
        let mut prev_in_root = self.in_root;
        if id != ast::CRATE_NODE_ID {
            prev_in_root = mem::replace(&mut self.in_root, false);
        }
        visit::walk_mod(self, m);
        self.in_root = prev_in_root;
    }

    fn visit_mac(&mut self, mac: &ast::Mac) {
        visit::walk_mac(self, mac)
    }
}

// Creates a new module which looks like:
//
//      mod $gensym {
//          extern crate rustc_macro;
//
//          use rustc_macro::__internal::Registry;
//
//          #[plugin_registrar]
//          fn registrar(registrar: &mut Registry) {
//              registrar.register_custom_derive($name_trait1, ::$name1);
//              registrar.register_custom_derive($name_trait2, ::$name2);
//              // ...
//          }
//      }
fn mk_registrar(cx: &mut ExtCtxt,
                custom_derives: &[CustomDerive]) -> P<ast::Item> {
    let eid = cx.codemap().record_expansion(ExpnInfo {
        call_site: DUMMY_SP,
        callee: NameAndSpan {
            format: MacroAttribute(token::intern("rustc_macro")),
            span: None,
            allow_internal_unstable: true,
        }
    });
    let span = Span { expn_id: eid, ..DUMMY_SP };

    let rustc_macro = token::str_to_ident("rustc_macro");
    let krate = cx.item(span,
                        rustc_macro,
                        Vec::new(),
                        ast::ItemKind::ExternCrate(None));

    let __internal = token::str_to_ident("__internal");
    let registry = token::str_to_ident("Registry");
    let registrar = token::str_to_ident("registrar");
    let register_custom_derive = token::str_to_ident("register_custom_derive");
    let stmts = custom_derives.iter().map(|cd| {
        let path = cx.path_global(cd.span, vec![cd.function_name]);
        let trait_name = cx.expr_str(cd.span, cd.trait_name.clone());
        (path, trait_name)
    }).map(|(path, trait_name)| {
        let registrar = cx.expr_ident(span, registrar);
        let ufcs_path = cx.path(span, vec![rustc_macro, __internal, registry,
                                           register_custom_derive]);
        cx.expr_call(span,
                     cx.expr_path(ufcs_path),
                     vec![registrar, trait_name, cx.expr_path(path)])
    }).map(|expr| {
        cx.stmt_expr(expr)
    }).collect::<Vec<_>>();

    let path = cx.path(span, vec![rustc_macro, __internal, registry]);
    let registrar_path = cx.ty_path(path);
    let arg_ty = cx.ty_rptr(span, registrar_path, None, ast::Mutability::Mutable);
    let func = cx.item_fn(span,
                          registrar,
                          vec![cx.arg(span, registrar, arg_ty)],
                          cx.ty(span, ast::TyKind::Tup(Vec::new())),
                          cx.block(span, stmts));

    let derive_registrar = token::intern_and_get_ident("rustc_derive_registrar");
    let derive_registrar = cx.meta_word(span, derive_registrar);
    let derive_registrar = cx.attribute(span, derive_registrar);
    let func = func.map(|mut i| {
        i.attrs.push(derive_registrar);
        i.vis = ast::Visibility::Public;
        i
    });
    let module = cx.item_mod(span,
                             span,
                             ast::Ident::with_empty_ctxt(token::gensym("registrar")),
                             Vec::new(),
                             vec![krate, func]);
    module.map(|mut i| {
        i.vis = ast::Visibility::Public;
        i
    })
}
