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

use metadata::csearch;
use middle::ty::{ty_struct, ty_enum};
use middle::ty;
use middle::typeck::{method_map, method_origin, method_param, method_self};
use middle::typeck::{method_super};
use middle::typeck::{method_static, method_trait};

use core::util::ignore;
use syntax::ast::{decl_item, def, def_fn, def_id, def_static_method};
use syntax::ast::{def_variant, expr_field, expr_method_call, expr_path};
use syntax::ast::{expr_struct, expr_unary, ident, inherited, item_enum};
use syntax::ast::{item_foreign_mod, item_fn, item_impl, item_struct};
use syntax::ast::{item_trait, local_crate, node_id, pat_struct, Path};
use syntax::ast::{private, provided, public, required, stmt_decl, visibility};
use syntax::ast;
use syntax::ast_map::{node_foreign_item, node_item, node_method};
use syntax::ast_map::{node_trait_method};
use syntax::ast_map;
use syntax::ast_util::{Private, Public, is_local};
use syntax::ast_util::{variant_visibility_to_privacy, visibility_to_privacy};
use syntax::attr;
use syntax::codemap::span;
use syntax::visit;

pub fn check_crate(tcx: ty::ctxt,
                   method_map: &method_map,
                   crate: @ast::crate) {
    let privileged_items = @mut ~[];

    // Adds an item to its scope.
    let add_privileged_item: @fn(@ast::item, &mut uint) = |item, count| {
        match item.node {
            item_struct(*) | item_trait(*) | item_enum(*) |
            item_fn(*) => {
                privileged_items.push(item.id);
                *count += 1;
            }
            item_impl(_, _, _, ref methods) => {
                for methods.each |method| {
                    privileged_items.push(method.id);
                    *count += 1;
                }
                privileged_items.push(item.id);
                *count += 1;
            }
            item_foreign_mod(ref foreign_mod) => {
                for foreign_mod.items.each |foreign_item| {
                    privileged_items.push(foreign_item.id);
                    *count += 1;
                }
            }
            _ => {}
        }
    };

    // Adds items that are privileged to this scope.
    let add_privileged_items: @fn(&[@ast::item]) -> uint = |items| {
        let mut count = 0;
        for items.each |&item| {
            add_privileged_item(item, &mut count);
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
            visibility_to_privacy(parent_vis)
        }
        else {
            // WRONG
            Public
        };
        debug!("parental_privacy = %?", parental_privacy);
        debug!("vis = %?, priv = %?",
               variant_info.vis,
               visibility_to_privacy(variant_info.vis))
        // inherited => privacy of the enum item
        if variant_visibility_to_privacy(variant_info.vis,
                                         parental_privacy == Public)
                                         == Private {
            tcx.sess.span_err(span,
                "can only dereference enums \
                 with a single, public variant");
        }
    };

    // Returns the ID of the container (impl or trait) that a crate-local
    // method belongs to.
    let local_method_container_id:
            @fn(span: span, method_id: node_id) -> def_id =
            |span, method_id| {
        match tcx.items.find(&method_id) {
            Some(&node_method(_, impl_id, _)) => impl_id,
            Some(&node_trait_method(_, trait_id, _)) => trait_id,
            Some(_) => {
                tcx.sess.span_bug(span,
                                  fmt!("method was a %s?!",
                                       ast_map::node_id_to_str(
                                            tcx.items,
                                            method_id,
                                            tcx.sess.parse_sess.interner)));
            }
            None => {
                tcx.sess.span_bug(span, "method not found in \
                                         AST map?!");
            }
        }
    };

    // Returns true if a crate-local method is private and false otherwise.
    let method_is_private: @fn(span: span, method_id: node_id) -> bool =
            |span, method_id| {
        let check = |vis: visibility, container_id: def_id| {
            let mut is_private = false;
            if vis == private {
                is_private = true;
            } else if vis == public {
                is_private = false;
            } else {
                // Look up the enclosing impl.
                if container_id.crate != local_crate {
                    tcx.sess.span_bug(span,
                                      "local method isn't in local \
                                       impl?!");
                }

                match tcx.items.find(&container_id.node) {
                    Some(&node_item(item, _)) => {
                        match item.node {
                            item_impl(_, None, _, _)
                                    if item.vis != public => {
                                is_private = true;
                            }
                            _ => {}
                        }
                    }
                    Some(_) => {
                        tcx.sess.span_bug(span, "impl wasn't an item?!");
                    }
                    None => {
                        tcx.sess.span_bug(span, "impl wasn't in AST map?!");
                    }
                }
            }

            is_private
        };

        match tcx.items.find(&method_id) {
            Some(&node_method(method, impl_id, _)) => {
                check(method.vis, impl_id)
            }
            Some(&node_trait_method(trait_method, trait_id, _)) => {
                match *trait_method {
                    required(_) => check(public, trait_id),
                    provided(method) => check(method.vis, trait_id),
                }
            }
            Some(_) => {
                tcx.sess.span_bug(span,
                                  fmt!("method_is_private: method was a %s?!",
                                       ast_map::node_id_to_str(
                                            tcx.items,
                                            method_id,
                                            tcx.sess.parse_sess.interner)));
            }
            None => {
                tcx.sess.span_bug(span, "method not found in \
                                         AST map?!");
            }
        }
    };

    // Returns true if the given local item is private and false otherwise.
    let local_item_is_private: @fn(span: span, item_id: node_id) -> bool =
            |span, item_id| {
        let mut f: &fn(node_id) -> bool = |_| false;
        f = |item_id| {
            match tcx.items.find(&item_id) {
                Some(&node_item(item, _)) => item.vis != public,
                Some(&node_foreign_item(_, _, vis, _)) => vis != public,
                Some(&node_method(method, impl_did, _)) => {
                    match method.vis {
                        private => true,
                        public => false,
                        inherited => f(impl_did.node)
                    }
                }
                Some(&node_trait_method(_, trait_did, _)) => f(trait_did.node),
                Some(_) => {
                    tcx.sess.span_bug(span,
                                      fmt!("local_item_is_private: item was \
                                            a %s?!",
                                           ast_map::node_id_to_str(
                                                tcx.items,
                                                item_id,
                                                tcx.sess
                                                   .parse_sess
                                                   .interner)));
                }
                None => {
                    tcx.sess.span_bug(span, "item not found in AST map?!");
                }
            }
        };
        f(item_id)
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

    // Given the ID of a method, checks to ensure it's in scope.
    let check_method_common: @fn(span: span,
                                 method_id: def_id,
                                 name: &ident) =
            |span, method_id, name| {
        if method_id.crate == local_crate {
            let is_private = method_is_private(span, method_id.node);
            let container_id = local_method_container_id(span,
                                                         method_id.node);
            if is_private &&
                    (container_id.crate != local_crate ||
                     !privileged_items.contains(&(container_id.node))) {
                tcx.sess.span_err(span,
                                  fmt!("method `%s` is private",
                                       *tcx.sess
                                           .parse_sess
                                           .interner
                                           .get(*name)));
            }
        } else {
            let visibility =
                csearch::get_item_visibility(tcx.sess.cstore, method_id);
            if visibility != public {
                tcx.sess.span_err(span,
                                  fmt!("method `%s` is private",
                                       *tcx.sess.parse_sess.interner
                                           .get(*name)));
            }
        }
    };

    // Checks that a private path is in scope.
    let check_path: @fn(span: span, def: def, path: @Path) =
            |span, def, path| {
        debug!("checking path");
        match def {
            def_static_method(method_id, _, _) => {
                debug!("found static method def, checking it");
                check_method_common(span, method_id, path.idents.last())
            }
            def_fn(def_id, _) => {
                if def_id.crate == local_crate {
                    if local_item_is_private(span, def_id.node) &&
                            !privileged_items.contains(&def_id.node) {
                        tcx.sess.span_err(span,
                                          fmt!("function `%s` is private",
                                               *tcx.sess
                                                   .parse_sess
                                                   .interner
                                                   .get(copy *path
                                                             .idents
                                                             .last())));
                    }
                } else if csearch::get_item_visibility(tcx.sess.cstore,
                                                       def_id) != public {
                    tcx.sess.span_err(span,
                                      fmt!("function `%s` is private",
                                           *tcx.sess
                                               .parse_sess
                                               .interner
                                               .get(copy *path
                                                         .idents
                                                         .last())));
                }
            }
            _ => {}
        }
    };

    // Checks that a private method is in scope.
    let check_method: @fn(span: span,
                          origin: &method_origin,
                          ident: ast::ident) =
            |span, origin, ident| {
        match *origin {
            method_static(method_id) => {
                check_method_common(span, method_id, &ident)
            }
            method_param(method_param {
                trait_id: trait_id,
                 method_num: method_num,
                 _
            }) |
            method_trait(trait_id, method_num, _) |
            method_self(trait_id, method_num) |
            method_super(trait_id, method_num) => {
                if trait_id.crate == local_crate {
                    match tcx.items.find(&trait_id.node) {
                        Some(&node_item(item, _)) => {
                            match item.node {
                                item_trait(_, _, ref methods) => {
                                    if method_num >= (*methods).len() {
                                        tcx.sess.span_bug(span, "method \
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
                                    tcx.sess.span_bug(span, "trait wasn't \
                                                             actually a \
                                                             trait?!");
                                }
                            }
                        }
                        Some(_) => {
                            tcx.sess.span_bug(span, "trait wasn't an \
                                                     item?!");
                        }
                        None => {
                            tcx.sess.span_bug(span, "trait item wasn't \
                                                     found in the AST \
                                                     map?!");
                        }
                    }
                } else {
                    // FIXME #4732: External crates.
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
        visit_item: |item, method_map, visitor| {
            // Do not check privacy inside items with the resolve_unexported
            // attribute. This is used for the test runner.
            if !attr::contains_name(attr::attr_metas(/*bad*/copy item.attrs),
                                    "!resolve_unexported") {
                visit::visit_item(item, method_map, visitor);
            }
        },
        visit_block: |block, method_map, visitor| {
            // Gather up all the privileged items.
            let mut n_added = 0;
            for block.node.stmts.each |stmt| {
                match stmt.node {
                    stmt_decl(decl, _) => {
                        match decl.node {
                            decl_item(item) => {
                                add_privileged_item(item, &mut n_added);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }

            visit::visit_block(block, method_map, visitor);

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
                            match method_map.find(&expr.id) {
                                None => {
                                    debug!("(privacy checking) checking \
                                            field access");
                                    check_field(expr.span, id, ident);
                                }
                                Some(ref entry) => {
                                    debug!("(privacy checking) checking \
                                            impl method");
                                    check_method(expr.span,
                                                 &entry.origin,
                                                 ident);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                expr_method_call(base, ident, _, _, _) => {
                    // Ditto
                    match ty::get(ty::type_autoderef(tcx, ty::expr_ty(tcx,
                                                          base))).sty {
                        ty_struct(id, _)
                        if id.crate != local_crate ||
                           !privileged_items.contains(&(id.node)) => {
                            match method_map.find(&expr.id) {
                                None => {
                                    tcx.sess.span_bug(expr.span,
                                                      "method call not in \
                                                       method map");
                                }
                                Some(ref entry) => {
                                    debug!("(privacy checking) checking \
                                            impl method");
                                    check_method(expr.span,
                                                 &entry.origin,
                                                 ident);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                expr_path(path) => {
                    check_path(expr.span, tcx.def_map.get_copy(&expr.id), path);
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
                                match tcx.def_map.get_copy(&expr.id) {
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
                                                          "resolve didn't \
                                                           map enum struct \
                                                           constructor to a \
                                                           variant def");
                                    }
                                }
                            }
                        }
                        _ => {
                            tcx.sess.span_bug(expr.span, "struct expr \
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
            match pattern.node {
                pat_struct(_, ref fields, _) => {
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
                                match tcx.def_map.find(&pattern.id) {
                                    Some(&def_variant(_, variant_id)) => {
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
                                                          "resolve didn't \
                                                           map enum struct \
                                                           pattern to a \
                                                           variant def");
                                    }
                                }
                            }
                        }
                        _ => {
                            tcx.sess.span_bug(pattern.span,
                                              "struct pattern didn't have \
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
    visit::visit_crate(crate, method_map, visitor);
}
