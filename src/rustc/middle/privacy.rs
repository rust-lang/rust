// A pass that checks to make sure private fields and methods aren't used
// outside their scopes.

use /*mod*/ syntax::ast;
use /*mod*/ syntax::visit;
use syntax::ast::{expr_field, expr_struct, ident, item_class, item_impl};
use syntax::ast::{item_trait, local_crate, node_id, pat_struct, private};
use syntax::ast::{provided, required};
use syntax::ast_map::{node_item, node_method};
use ty::ty_class;
use typeck::{method_map, method_origin, method_param, method_self};
use typeck::{method_static, method_trait};

use core::util::ignore;
use dvec::DVec;

fn check_crate(tcx: ty::ctxt, method_map: &method_map, crate: @ast::crate) {
    let privileged_items = @DVec();

    // Adds structs that are privileged to this scope.
    let add_privileged_items = |items: &[@ast::item]| {
        let mut count = 0;
        for items.each |item| {
            match item.node {
                item_class(*) | item_trait(*) | item_impl(*) => {
                    privileged_items.push(item.id);
                    count += 1;
                }
                _ => {}
            }
        }
        count
    };

    // Checks that a private field is in scope.
    let check_field = |span, id, ident| {
        let fields = ty::lookup_class_fields(tcx, id);
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
    let check_method = |span, origin: &method_origin| {
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
            method_param({trait_id: trait_id, method_num: method_num, _}) |
            method_trait(trait_id, method_num, _) |
            method_self(trait_id, method_num) => {
                if trait_id.crate == local_crate {
                    match tcx.items.find(trait_id.node) {
                        Some(node_item(item, _)) => {
                            match item.node {
                                item_trait(_, _, methods) => {
                                    if method_num >= methods.len() {
                                        tcx.sess.span_bug(span, ~"method \
                                                                  number \
                                                                  out of \
                                                                  range?!");
                                    }
                                    match methods[method_num] {
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

    let visitor = visit::mk_vt(@{
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
                    match ty::get(ty::expr_ty(tcx, base)).sty {
                        ty_class(id, _)
                        if id.crate != local_crate ||
                           !privileged_items.contains(&(id.node)) => {
                            match method_map.find(expr.id) {
                                None => {
                                    debug!("(privacy checking) checking \
                                            field access");
                                    check_field(expr.span, id, ident);
                                }
                                Some(entry) => {
                                    debug!("(privacy checking) checking \
                                            impl method");
                                    check_method(expr.span, &entry.origin);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                expr_struct(_, fields, _) => {
                    match ty::get(ty::expr_ty(tcx, expr)).sty {
                        ty_class(id, _) => {
                            if id.crate != local_crate ||
                                    !privileged_items.contains(&(id.node)) {
                                for fields.each |field| {
                                        debug!("(privacy checking) checking \
                                                field in struct literal");
                                    check_field(expr.span, id,
                                                field.node.ident);
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
                _ => {}
            }

            visit::visit_expr(expr, method_map, visitor);
        },
        visit_pat: |pattern, method_map, visitor| {
            match pattern.node {
                pat_struct(_, fields, _) => {
                    match ty::get(ty::pat_ty(tcx, pattern)).sty {
                        ty_class(id, _) => {
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

