// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// A pass that checks to make sure private fields and methods aren't used
// outside their scopes.

use core::prelude::*;

use middle::ty::{ty_struct, ty_enum};
use middle::ty;
use middle::typeck::{method_map, method_origin, method_param, method_self};
use middle::typeck::{method_static, method_trait};

use core::dvec::DVec;
use core::util::ignore;
use syntax::ast::{def_variant, expr_field, expr_method_call, expr_struct};
use syntax::ast::{expr_unary, ident, item_struct, item_enum, item_impl};
use syntax::ast::{item_trait, local_crate, node_id, pat_struct, private};
use syntax::ast::{provided, required};
use syntax::ast;
use syntax::ast_map::{node_item, node_method};
use syntax::ast_map;
use syntax::ast_util::{Private, Public, has_legacy_export_attr, is_local};
use syntax::ast_util::{visibility_to_privacy};
use syntax::codemap::span;
use syntax::visit;

fn check_crate(tcx: ty::ctxt, method_map: &method_map, crate: @ast::crate) {
    let privileged_items = @DVec();
    let legacy_exports = has_legacy_export_attr(crate.node.attrs);

    // Adds structs that are privileged to this scope.
    let add_privileged_items: @fn(&[@ast::item]) -> int = |items| {
        let mut count = 0;
        for items.each |item| {
            match item.node {
                item_struct(*) | item_trait(*) | item_impl(*)
                | item_enum(*) => {
                    privileged_items.push(item.id);
                    count += 1;
                }
                _ => {}
            }
        }
        count
    };

    // Checks that an enum variant is in scope
    let check_variant: @fn(span: span, enum_id: ast::def_id) =
            |span, enum_id| {
        let variant_info = ty::enum_variants(tcx, enum_id)[0];
        let parental_privacy = if is_local(enum_id) {
            let parent_vis = ast_map::node_item_query(tcx.items, enum_id.node,
                                   |it| { it.vis },
                                   ~"unbound enum parent when checking \
                                    dereference of enum type");
            visibility_to_privacy(parent_vis, legacy_exports)
        }
        else {
            // WRONG
            Public
        };
        debug!("parental_privacy = %?", parental_privacy);
        debug!("vis = %?, priv = %?, legacy_exports = %?",
               variant_info.vis,
               visibility_to_privacy(variant_info.vis, legacy_exports),
               legacy_exports);
        // inherited => privacy of the enum item
        if visibility_to_privacy(variant_info.vis,
                                 parental_privacy == Public) == Private {
            tcx.sess.span_err(span,
                ~"can only dereference enums \
                  with a single, public variant");
        }
    };

    // Checks that a private field is in scope.
    let check_field: @fn(span: span, id: ast::def_id, ident: ast::ident) =
            |span, id, ident| {
        let fields = ty::lookup_struct_fields(tcx, id);
        for fields.each |field| {
            if field.ident != ident { loop; }
            if field.vis == private {
                tcx.sess.span_err(span, fmt!("field `%s` is private",
                                             *tcx.sess.parse_sess.interner
                                                 .get(ident)));
            }
            break;
        }
    };

    // Checks that a private method is in scope.
    let check_method: @fn(span: span, origin: &method_origin) =
            |span, origin| {
        match *origin {
            method_static(method_id) => {
                if method_id.crate == local_crate {
                    match tcx.items.find(method_id.node) {
                        Some(node_method(method, impl_id, _)) => {
                            if method.vis == private &&
                                    (impl_id.crate != local_crate ||
                                     !privileged_items
                                     .contains(&(impl_id.node))) {
                                tcx.sess.span_err(span,
                                                  fmt!("method `%s` is \
                                                        private",
                                                       *tcx.sess
                                                           .parse_sess
                                                           .interner
                                                           .get(method
                                                                .ident)));
                            }
                        }
                        Some(_) => {
                            tcx.sess.span_bug(span, ~"method wasn't \
                                                      actually a method?!");
                        }
                        None => {
                            tcx.sess.span_bug(span, ~"method not found in \
                                                      AST map?!");
                        }
                    }
                } else {
                    // XXX: External crates.
                }
            }
            method_param(method_param {
                trait_id: trait_id,
                 method_num: method_num,
                 _
            }) |
            method_trait(trait_id, method_num, _) |
            method_self(trait_id, method_num) => {
                if trait_id.crate == local_crate {
                    match tcx.items.find(trait_id.node) {
                        Some(node_item(item, _)) => {
                            match item.node {
                                item_trait(_, _, ref methods) => {
                                    if method_num >= (*methods).len() {
                                        tcx.sess.span_bug(span, ~"method \
                                                                  number \
                                                                  out of \
                                                                  range?!");
                                    }
                                    match (*methods)[method_num] {
                                        provided(method)
                                             if method.vis == private &&
                                             !privileged_items
                                             .contains(&(trait_id.node)) => {
                                            tcx.sess.span_err(span,
                                                              fmt!("method
                                                                    `%s` \
                                                                    is \
                                                                    private",
                                                                   *tcx
                                                                   .sess
                                                                   .parse_sess
                                                                   .interner
                                                                   .get
                                                                   (method
                                                                    .ident)));
                                        }
                                        provided(_) | required(_) => {
                                            // Required methods can't be
                                            // private.
                                        }
                                    }
                                }
                                _ => {
                                    tcx.sess.span_bug(span, ~"trait wasn't \
                                                              actually a \
                                                              trait?!");
                                }
                            }
                        }
                        Some(_) => {
                            tcx.sess.span_bug(span, ~"trait wasn't an \
                                                      item?!");
                        }
                        None => {
                            tcx.sess.span_bug(span, ~"trait item wasn't \
                                                      found in the AST \
                                                      map?!");
                        }
                    }
                } else {
                    // XXX: External crates.
                }
            }
        }
    };

    let visitor = visit::mk_vt(@visit::Visitor {
        visit_mod: |the_module, span, node_id, method_map, visitor| {
            let n_added = add_privileged_items(the_module.items);

            visit::visit_mod(the_module, span, node_id, method_map, visitor);

            for n_added.times {
                ignore(privileged_items.pop());
            }
        },
        visit_expr: |expr, method_map: &method_map, visitor| {
            match expr.node {
                expr_field(base, ident, _) => {
                    // With type_autoderef, make sure we don't
                    // allow pointers to violate privacy
                    match ty::get(ty::type_autoderef(tcx, ty::expr_ty(tcx,
                                                          base))).sty {
                        ty_struct(id, _)
                        if id.crate != local_crate ||
                           !privileged_items.contains(&(id.node)) => {
                            match method_map.find(expr.id) {
                                None => {
                                    debug!("(privacy checking) checking \
                                            field access");
                                    check_field(expr.span, id, ident);
                                }
                                Some(ref entry) => {
                                    debug!("(privacy checking) checking \
                                            impl method");
                                    check_method(expr.span, &(*entry).origin);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                expr_method_call(base, _, _, _, _) => {
                    // Ditto
                    match ty::get(ty::type_autoderef(tcx, ty::expr_ty(tcx,
                                                          base))).sty {
                        ty_struct(id, _)
                        if id.crate != local_crate ||
                           !privileged_items.contains(&(id.node)) => {
                            match method_map.find(expr.id) {
                                None => {
                                    tcx.sess.span_bug(expr.span,
                                                      ~"method call not in \
                                                        method map");
                                }
                                Some(ref entry) => {
                                    debug!("(privacy checking) checking \
                                            impl method");
                                    check_method(expr.span, &(*entry).origin);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                expr_struct(_, ref fields, _) => {
                    match ty::get(ty::expr_ty(tcx, expr)).sty {
                        ty_struct(id, _) => {
                            if id.crate != local_crate ||
                                    !privileged_items.contains(&(id.node)) {
                                for (*fields).each |field| {
                                        debug!("(privacy checking) checking \
                                                field in struct literal");
                                    check_field(expr.span, id,
                                                field.node.ident);
                                }
                            }
                        }
                        ty_enum(id, _) => {
                            if id.crate != local_crate ||
                                    !privileged_items.contains(&(id.node)) {
                                match tcx.def_map.get(expr.id) {
                                    def_variant(_, variant_id) => {
                                        for (*fields).each |field| {
                                                debug!("(privacy checking) \
                                                        checking field in \
                                                        struct variant \
                                                        literal");
                                            check_field(expr.span, variant_id,
                                                        field.node.ident);
                                        }
                                    }
                                    _ => {
                                        tcx.sess.span_bug(expr.span,
                                                          ~"resolve didn't \
                                                            map enum struct \
                                                            constructor to a \
                                                            variant def");
                                    }
                                }
                            }
                        }
                        _ => {
                            tcx.sess.span_bug(expr.span, ~"struct expr \
                                                           didn't have \
                                                           struct type?!");
                        }
                    }
                }
                expr_unary(ast::deref, operand) => {
                    // In *e, we need to check that if e's type is an
                    // enum type t, then t's first variant is public or
                    // privileged. (We can assume it has only one variant
                    // since typeck already happened.)
                    match ty::get(ty::expr_ty(tcx, operand)).sty {
                        ty_enum(id, _) => {
                            if id.crate != local_crate ||
                                !privileged_items.contains(&(id.node)) {
                                check_variant(expr.span, id);
                            }
                        }
                        _ => { /* No check needed */ }
                    }
                }
                _ => {}
            }

            visit::visit_expr(expr, method_map, visitor);
        },
        visit_pat: |pattern, method_map, visitor| {
            match /*bad*/copy pattern.node {
                pat_struct(_, fields, _) => {
                    match ty::get(ty::pat_ty(tcx, pattern)).sty {
                        ty_struct(id, _) => {
                            if id.crate != local_crate ||
                                    !privileged_items.contains(&(id.node)) {
                                for fields.each |field| {
                                        debug!("(privacy checking) checking \
                                                struct pattern");
                                    check_field(pattern.span, id,
                                                field.ident);
                                }
                            }
                        }
                        ty_enum(enum_id, _) => {
                            if enum_id.crate != local_crate ||
                                    !privileged_items.contains(
                                        &enum_id.node) {
                                match tcx.def_map.find(pattern.id) {
                                    Some(def_variant(_, variant_id)) => {
                                        for fields.each |field| {
                                            debug!("(privacy checking) \
                                                    checking field in \
                                                    struct variant pattern");
                                            check_field(pattern.span,
                                                        variant_id,
                                                        field.ident);
                                        }
                                    }
                                    _ => {
                                        tcx.sess.span_bug(pattern.span,
                                                          ~"resolve didn't \
                                                            map enum struct \
                                                            pattern to a \
                                                            variant def");
                                    }
                                }
                            }
                        }
                        _ => {
                            tcx.sess.span_bug(pattern.span,
                                              ~"struct pattern didn't have \
                                                struct type?!");
                        }
                    }
                }
                _ => {}
            }

            visit::visit_pat(pattern, method_map, visitor);
        },
        .. *visit::default_visitor()
    });
    visit::visit_crate(*crate, method_map, visitor);
}

