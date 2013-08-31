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
use syntax::ast_map;
use syntax::attr;
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::Visitor;
use util::ppaux::Repr;

#[deriving(Clone)]
struct Context {
    tcx: ty::ctxt,
    safe_stack: bool
}

struct StackCheckVisitor;

impl Visitor<Context> for StackCheckVisitor {
    fn visit_item(&mut self, i:@ast::item, e:Context) {
        stack_check_item(*self, i, e);
    }
    fn visit_fn(&mut self, fk:&visit::fn_kind, fd:&ast::fn_decl,
                b:&ast::Block, s:Span, n:ast::NodeId, e:Context) {
        stack_check_fn(*self, fk, fd, b, s, n, e);
    }
    fn visit_expr(&mut self, ex:@ast::expr, e:Context) {
        stack_check_expr(*self, ex, e);
    }
}

pub fn stack_check_crate(tcx: ty::ctxt,
                         crate: &ast::Crate) {
    let new_cx = Context {
        tcx: tcx,
        safe_stack: false
    };
    let mut visitor = StackCheckVisitor;
    visit::walk_crate(&mut visitor, crate, new_cx);
}

fn stack_check_item(v: StackCheckVisitor,
                    item: @ast::item,
                    in_cx: Context) {
    let mut v = v;
    match item.node {
        ast::item_fn(_, ast::extern_fn, _, _, _) => {
            // an extern fn is already being called from C code...
            let new_cx = Context {safe_stack: true, ..in_cx};
            visit::walk_item(&mut v, item, new_cx);
        }
        ast::item_fn(*) => {
            let safe_stack = fixed_stack_segment(item.attrs);
            let new_cx = Context {safe_stack: safe_stack, ..in_cx};
            visit::walk_item(&mut v, item, new_cx);
        }
        ast::item_impl(_, _, _, ref methods) => {
            // visit_method() would make this nicer
            for &method in methods.iter() {
                let safe_stack = fixed_stack_segment(method.attrs);
                let new_cx = Context {safe_stack: safe_stack, ..in_cx};
                visit::walk_method_helper(&mut v, method, new_cx);
            }
        }
        _ => {
            visit::walk_item(&mut v, item, in_cx);
        }
    }

    fn fixed_stack_segment(attrs: &[ast::Attribute]) -> bool {
        attr::contains_name(attrs, "fixed_stack_segment")
    }
}

fn stack_check_fn<'a>(v: StackCheckVisitor,
                      fk: &visit::fn_kind,
                      decl: &ast::fn_decl,
                      body: &ast::Block,
                      sp: Span,
                      id: ast::NodeId,
                      in_cx: Context) {
    let safe_stack = match *fk {
        visit::fk_method(*) | visit::fk_item_fn(*) => {
            in_cx.safe_stack // see stack_check_item above
        }
        visit::fk_anon(*) | visit::fk_fn_block => {
            match ty::get(ty::node_id_to_type(in_cx.tcx, id)).sty {
                ty::ty_bare_fn(*) |
                ty::ty_closure(ty::ClosureTy {sigil: ast::OwnedSigil, _}) |
                ty::ty_closure(ty::ClosureTy {sigil: ast::ManagedSigil, _}) => {
                    false
                }
                _ => {
                    in_cx.safe_stack
                }
            }
        }
    };
    let new_cx = Context {safe_stack: safe_stack, ..in_cx};
    debug!("stack_check_fn(safe_stack=%b, id=%?)", safe_stack, id);
    let mut v = v;
    visit::walk_fn(&mut v, fk, decl, body, sp, id, new_cx);
}

fn stack_check_expr<'a>(v: StackCheckVisitor,
                        expr: @ast::expr,
                        cx: Context) {
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
                            call_to_extern_fn(cx, callee);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let mut v = v;
    visit::walk_expr(&mut v, expr, cx);
}

fn call_to_extern_fn(cx: Context, callee: @ast::expr) {
    // Permit direct calls to extern fns that are annotated with
    // #[rust_stack]. This is naturally a horrible pain to achieve.
    match callee.node {
        ast::expr_path(*) => {
            match cx.tcx.def_map.find(&callee.id) {
                Some(&ast::def_fn(id, _)) if id.crate == ast::LOCAL_CRATE => {
                    match cx.tcx.items.find(&id.node) {
                        Some(&ast_map::node_foreign_item(item, _, _, _)) => {
                            if attr::contains_name(item.attrs, "rust_stack") {
                                return;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }

    cx.tcx.sess.add_lint(lint::cstack,
                         callee.id,
                         callee.span,
                         fmt!("invoking non-Rust fn in fn without \
                              #[fixed_stack_segment]"));
}
