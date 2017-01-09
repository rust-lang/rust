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
use syntax::ext::base::ExtCtxt;
use syntax::ext::build::AstBuilder;
use syntax::ext::expand::ExpansionConfig;
use syntax::fold::Folder;
use syntax::parse::ParseSess;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax::visit::{self, Visitor};

use syntax_pos::{Span, DUMMY_SP};

use deriving;

struct CustomDerive {
    trait_name: ast::Name,
    function_name: Ident,
    span: Span,
    attrs: Vec<ast::Name>,
}

struct AttrProcMacro {
    function_name: Ident,
    span: Span,
}

struct CollectProcMacros<'a> {
    derives: Vec<CustomDerive>,
    attr_macros: Vec<AttrProcMacro>,
    in_root: bool,
    handler: &'a errors::Handler,
    is_proc_macro_crate: bool,
    is_test_crate: bool,
}

pub fn modify(sess: &ParseSess,
              resolver: &mut ::syntax::ext::base::Resolver,
              mut krate: ast::Crate,
              is_proc_macro_crate: bool,
              is_test_crate: bool,
              num_crate_types: usize,
              handler: &errors::Handler) -> ast::Crate {
    let ecfg = ExpansionConfig::default("proc_macro".to_string());
    let mut cx = ExtCtxt::new(sess, ecfg, resolver);

    let (derives, attr_macros) = {
        let mut collect = CollectProcMacros {
            derives: Vec::new(),
            attr_macros: Vec::new(),
            in_root: true,
            handler: handler,
            is_proc_macro_crate: is_proc_macro_crate,
            is_test_crate: is_test_crate,
        };
        visit::walk_crate(&mut collect, &krate);
        (collect.derives, collect.attr_macros)
    };

    if !is_proc_macro_crate {
        return krate
    }

    if num_crate_types > 1 {
        handler.err("cannot mix `proc-macro` crate type with others");
    }

    if is_test_crate {
        return krate;
    }

    krate.module.items.push(mk_registrar(&mut cx, &derives, &attr_macros));

    if krate.exported_macros.len() > 0 {
        handler.err("cannot export macro_rules! macros from a `proc-macro` \
                     crate type currently");
    }

    return krate
}

impl<'a> CollectProcMacros<'a> {
    fn check_not_pub_in_root(&self, vis: &ast::Visibility, sp: Span) {
        if self.is_proc_macro_crate &&
           self.in_root &&
           *vis == ast::Visibility::Public {
            self.handler.span_err(sp,
                                  "`proc-macro` crate types cannot \
                                   export any items other than functions \
                                   tagged with `#[proc_macro_derive]` currently");
        }
    }

    fn collect_custom_derive(&mut self, item: &'a ast::Item, attr: &'a ast::Attribute) {
        // Once we've located the `#[proc_macro_derive]` attribute, verify
        // that it's of the form `#[proc_macro_derive(Foo)]` or
        // `#[proc_macro_derive(Foo, attributes(A, ..))]`
        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => {
                self.handler.span_err(attr.span(),
                                      "attribute must be of form: \
                                       #[proc_macro_derive(TraitName)]");
                return
            }
        };
        if list.len() != 1 && list.len() != 2 {
            self.handler.span_err(attr.span(),
                                  "attribute must have either one or two arguments");
            return
        }
        let trait_attr = &list[0];
        let attributes_attr = list.get(1);
        let trait_name = match trait_attr.name() {
            Some(name) => name,
            _ => {
                self.handler.span_err(trait_attr.span(), "not a meta item");
                return
            }
        };
        if !trait_attr.is_word() {
            self.handler.span_err(trait_attr.span(), "must only be one word");
        }

        if deriving::is_builtin_trait(trait_name) {
            self.handler.span_err(trait_attr.span(),
                                  "cannot override a built-in #[derive] mode");
        }

        if self.derives.iter().any(|d| d.trait_name == trait_name) {
            self.handler.span_err(trait_attr.span(),
                                  "derive mode defined twice in this crate");
        }

        let proc_attrs: Vec<_> = if let Some(attr) = attributes_attr {
            if !attr.check_name("attributes") {
                self.handler.span_err(attr.span(), "second argument must be `attributes`")
            }
            attr.meta_item_list().unwrap_or_else(|| {
                self.handler.span_err(attr.span(),
                                      "attribute must be of form: \
                                       `attributes(foo, bar)`");
                &[]
            }).into_iter().filter_map(|attr| {
                let name = match attr.name() {
                    Some(name) => name,
                    _ => {
                        self.handler.span_err(attr.span(), "not a meta item");
                        return None;
                    },
                };

                if !attr.is_word() {
                    self.handler.span_err(attr.span(), "must only be one word");
                    return None;
                }

                Some(name)
            }).collect()
        } else {
            Vec::new()
        };

        if self.in_root && item.vis == ast::Visibility::Public {
            self.derives.push(CustomDerive {
                span: item.span,
                trait_name: trait_name,
                function_name: item.ident,
                attrs: proc_attrs,
            });
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro_derive]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro_derive]` must be `pub`"
            };
            self.handler.span_err(item.span, msg);
        }
    }

    fn collect_attr_proc_macro(&mut self, item: &'a ast::Item, attr: &'a ast::Attribute) {
        if let Some(_) = attr.meta_item_list() {
            self.handler.span_err(attr.span, "`#[proc_macro_attribute]` attribute
                cannot contain any meta items");
            return;
        }

        if self.in_root && item.vis == ast::Visibility::Public {
            self.attr_macros.push(AttrProcMacro {
                span: item.span,
                function_name: item.ident,
            });
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro_attribute]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro_attribute]` must be `pub`"
            };
            self.handler.span_err(item.span, msg);
        }
    }
}

impl<'a> Visitor<'a> for CollectProcMacros<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        // First up, make sure we're checking a bare function. If we're not then
        // we're just not interested in this item.
        //
        // If we find one, try to locate a `#[proc_macro_derive]` attribute on
        // it.
        let is_fn = match item.node {
            ast::ItemKind::Fn(..) => true,
            _ => false,
        };

        let mut found_attr: Option<&'a ast::Attribute> = None;

        for attr in &item.attrs {
            if attr.check_name("proc_macro_derive") || attr.check_name("proc_macro_attribute") {
                if let Some(prev_attr) = found_attr {
                    let msg = if attr.name() == prev_attr.name() {
                        format!("Only one `#[{}]` attribute is allowed on any given function",
                                attr.name())
                    } else {
                        format!("`#[{}]` and `#[{}]` attributes cannot both be applied \
                                to the same function", attr.name(), prev_attr.name())
                    };

                    self.handler.struct_span_err(attr.span(), &msg)
                        .span_note(prev_attr.span(), "Previous attribute here")
                        .emit();

                    return;
                }

                found_attr = Some(attr);
            }
        }

        let attr = match found_attr {
            None => {
                self.check_not_pub_in_root(&item.vis, item.span);
                return visit::walk_item(self, item);
            },
            Some(attr) => attr,
        };

        if !is_fn {
            let msg = format!("the `#[{}]` attribute may only be used on bare functions",
                              attr.name());

            self.handler.span_err(attr.span(), &msg);
            return;
        }

        if self.is_test_crate {
            return;
        }

        if !self.is_proc_macro_crate {
            let msg = format!("the `#[{}]` attribute is only usable with crates of the \
                              `proc-macro` crate type", attr.name());

            self.handler.span_err(attr.span(), &msg);
            return;
        }

        if attr.check_name("proc_macro_derive") {
            self.collect_custom_derive(item, attr);
        } else if attr.check_name("proc_macro_attribute") {
            self.collect_attr_proc_macro(item, attr);
        };

        visit::walk_item(self, item);
    }

    fn visit_mod(&mut self, m: &'a ast::Mod, _s: Span, id: NodeId) {
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
//          extern crate proc_macro;
//
//          use proc_macro::__internal::Registry;
//
//          #[plugin_registrar]
//          fn registrar(registrar: &mut Registry) {
//              registrar.register_custom_derive($name_trait1, ::$name1, &[]);
//              registrar.register_custom_derive($name_trait2, ::$name2, &["attribute_name"]);
//              // ...
//          }
//      }
fn mk_registrar(cx: &mut ExtCtxt,
                custom_derives: &[CustomDerive],
                custom_attrs: &[AttrProcMacro]) -> P<ast::Item> {
    let eid = cx.codemap().record_expansion(ExpnInfo {
        call_site: DUMMY_SP,
        callee: NameAndSpan {
            format: MacroAttribute(Symbol::intern("proc_macro")),
            span: None,
            allow_internal_unstable: true,
        }
    });
    let span = Span { expn_id: eid, ..DUMMY_SP };

    let proc_macro = Ident::from_str("proc_macro");
    let krate = cx.item(span,
                        proc_macro,
                        Vec::new(),
                        ast::ItemKind::ExternCrate(None));

    let __internal = Ident::from_str("__internal");
    let registry = Ident::from_str("Registry");
    let registrar = Ident::from_str("registrar");
    let register_custom_derive = Ident::from_str("register_custom_derive");
    let register_attr_proc_macro = Ident::from_str("register_attr_proc_macro");

    let mut stmts = custom_derives.iter().map(|cd| {
        let path = cx.path_global(cd.span, vec![cd.function_name]);
        let trait_name = cx.expr_str(cd.span, cd.trait_name);
        let attrs = cx.expr_vec_slice(
            span,
            cd.attrs.iter().map(|&s| cx.expr_str(cd.span, s)).collect::<Vec<_>>()
        );
        let registrar = cx.expr_ident(span, registrar);
        let ufcs_path = cx.path(span, vec![proc_macro, __internal, registry,
                                           register_custom_derive]);

        cx.stmt_expr(cx.expr_call(span, cx.expr_path(ufcs_path),
                                  vec![registrar, trait_name, cx.expr_path(path), attrs]))

    }).collect::<Vec<_>>();

    stmts.extend(custom_attrs.iter().map(|ca| {
        let name = cx.expr_str(ca.span, ca.function_name.name);
        let path = cx.path_global(ca.span, vec![ca.function_name]);
        let registrar = cx.expr_ident(ca.span, registrar);

        let ufcs_path = cx.path(span,
                                vec![proc_macro, __internal, registry, register_attr_proc_macro]);

        cx.stmt_expr(cx.expr_call(span, cx.expr_path(ufcs_path),
                                  vec![registrar, name, cx.expr_path(path)]))
    }));

    let path = cx.path(span, vec![proc_macro, __internal, registry]);
    let registrar_path = cx.ty_path(path);
    let arg_ty = cx.ty_rptr(span, registrar_path, None, ast::Mutability::Mutable);
    let func = cx.item_fn(span,
                          registrar,
                          vec![cx.arg(span, registrar, arg_ty)],
                          cx.ty(span, ast::TyKind::Tup(Vec::new())),
                          cx.block(span, stmts));

    let derive_registrar = cx.meta_word(span, Symbol::intern("rustc_derive_registrar"));
    let derive_registrar = cx.attribute(span, derive_registrar);
    let func = func.map(|mut i| {
        i.attrs.push(derive_registrar);
        i.vis = ast::Visibility::Public;
        i
    });
    let ident = ast::Ident::with_empty_ctxt(Symbol::gensym("registrar"));
    let module = cx.item_mod(span, span, ident, Vec::new(), vec![krate, func]).map(|mut i| {
        i.vis = ast::Visibility::Public;
        i
    });

    cx.monotonic_expander().fold_item(module).pop().unwrap()
}
