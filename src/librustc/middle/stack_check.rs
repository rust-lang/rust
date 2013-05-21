// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Lint mode to detect cases where we call non-Rust fns, which do not
have a stack growth check, from locations not annotated to request
large stacks.

*/

use middle::lint;
use middle::ty;
use syntax::ast;
use syntax::attr;
use syntax::codemap::span;
use visit = syntax::oldvisit;
use util::ppaux::Repr;

#[deriving(Clone)]
struct Context {
    tcx: ty::ctxt,
    safe_stack: bool
}

pub fn stack_check_crate(tcx: ty::ctxt,
                         crate: &ast::Crate) {
    let new_cx = Context {
        tcx: tcx,
        safe_stack: false
    };
    let visitor = visit::mk_vt(@visit::Visitor {
        visit_item: stack_check_item,
        visit_fn: stack_check_fn,
        visit_expr: stack_check_expr,
        ..*visit::default_visitor()
    });
    visit::visit_crate(crate, (new_cx, visitor));
}

fn stack_check_item(item: @ast::item,
                    (in_cx, v): (Context, visit::vt<Context>)) {
    let safe_stack = match item.node {
        ast::item_fn(*) => {
            attr::contains_name(item.attrs, "fixed_stack_segment")
        }
        _ => {
            false
        }
    };
    let new_cx = Context {
        tcx: in_cx.tcx,
        safe_stack: safe_stack
    };
    visit::visit_item(item, (new_cx, v));
}

fn stack_check_fn<'a>(fk: &visit::fn_kind,
                      decl: &ast::fn_decl,
                      body: &ast::Block,
                      sp: span,
                      id: ast::NodeId,
                      (in_cx, v): (Context, visit::vt<Context>)) {
    let safe_stack = match *fk {
        visit::fk_item_fn(*) => in_cx.safe_stack, // see stack_check_item above
        visit::fk_anon(*) | visit::fk_fn_block | visit::fk_method(*) => false,
    };
    let new_cx = Context {
        tcx: in_cx.tcx,
        safe_stack: safe_stack
    };
    debug!("stack_check_fn(safe_stack=%b, id=%?)", safe_stack, id);
    visit::visit_fn(fk, decl, body, sp, id, (new_cx, v));
}

fn stack_check_expr<'a>(expr: @ast::expr,
                        (cx, v): (Context, visit::vt<Context>)) {
    debug!("stack_check_expr(safe_stack=%b, expr=%s)",
           cx.safe_stack, expr.repr(cx.tcx));
    if !cx.safe_stack {
        match expr.node {
            ast::expr_call(callee, _, _) => {
                let callee_ty = ty::expr_ty(cx.tcx, callee);
                debug!("callee_ty=%s", callee_ty.repr(cx.tcx));
                match ty::get(callee_ty).sty {
                    ty::ty_bare_fn(ref fty) => {
                        if !fty.abis.is_rust() && !fty.abis.is_intrinsic() {
                            cx.tcx.sess.add_lint(
                                lint::cstack,
                                callee.id,
                                callee.span,
                                fmt!("invoking non-Rust fn in fn without \
                                      #[fixed_stack_segment]"));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    visit::visit_expr(expr, (cx, v));
}
