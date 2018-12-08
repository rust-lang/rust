// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir;
use crate::rustc::hir::def::Def;
use crate::rustc::hir::intravisit;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::ty;
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_data_structures::fx::FxHashSet;
use crate::rustc_target::spec::abi::Abi;
use crate::syntax::ast;
use crate::syntax::source_map::Span;
use crate::utils::{iter_input_pats, span_lint, type_is_unsafe_function};
use matches::matches;

/// **What it does:** Checks for functions with too many parameters.
///
/// **Why is this bad?** Functions with lots of parameters are considered bad
/// style and reduce readability (“what does the 5th parameter mean?”). Consider
/// grouping some parameters into a new type.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) {
///     ..
/// }
/// ```
declare_clippy_lint! {
    pub TOO_MANY_ARGUMENTS,
    complexity,
    "functions with too many arguments"
}

/// **What it does:** Checks for public functions that dereferences raw pointer
/// arguments but are not marked unsafe.
///
/// **Why is this bad?** The function should probably be marked `unsafe`, since
/// for an arbitrary raw pointer, there is no way of telling for sure if it is
/// valid.
///
/// **Known problems:**
///
/// * It does not check functions recursively so if the pointer is passed to a
/// private non-`unsafe` function which does the dereferencing, the lint won't
/// trigger.
/// * It only checks for arguments whose type are raw pointers, not raw pointers
/// got from an argument in some other way (`fn foo(bar: &[*const u8])` or
/// `some_argument.get_raw_ptr()`).
///
/// **Example:**
/// ```rust
/// pub fn foo(x: *const u8) {
///     println!("{}", unsafe { *x });
/// }
/// ```
declare_clippy_lint! {
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    correctness,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
}

#[derive(Copy, Clone)]
pub struct Functions {
    threshold: u64,
}

impl Functions {
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }
}

impl LintPass for Functions {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOO_MANY_ARGUMENTS, NOT_UNSAFE_PTR_ARG_DEREF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Functions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        span: Span,
        nodeid: ast::NodeId,
    ) {
        let is_impl = if let Some(hir::Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(nodeid)) {
            matches!(item.node, hir::ItemKind::Impl(_, _, _, _, Some(_), _, _))
        } else {
            false
        };

        let unsafety = match kind {
            hir::intravisit::FnKind::ItemFn(_, _, hir::FnHeader { unsafety, .. }, _, _) => unsafety,
            hir::intravisit::FnKind::Method(_, sig, _, _) => sig.header.unsafety,
            hir::intravisit::FnKind::Closure(_) => return,
        };

        // don't warn for implementations, it's not their fault
        if !is_impl {
            // don't lint extern functions decls, it's not their fault either
            match kind {
                hir::intravisit::FnKind::Method(
                    _,
                    &hir::MethodSig {
                        header: hir::FnHeader { abi: Abi::Rust, .. },
                        ..
                    },
                    _,
                    _,
                )
                | hir::intravisit::FnKind::ItemFn(_, _, hir::FnHeader { abi: Abi::Rust, .. }, _, _) => {
                    self.check_arg_number(cx, decl, span)
                },
                _ => {},
            }
        }

        self.check_raw_ptr(cx, unsafety, decl, body, nodeid);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::TraitItem) {
        if let hir::TraitItemKind::Method(ref sig, ref eid) = item.node {
            // don't lint extern functions decls, it's not their fault
            if sig.header.abi == Abi::Rust {
                self.check_arg_number(cx, &sig.decl, item.span);
            }

            if let hir::TraitMethod::Provided(eid) = *eid {
                let body = cx.tcx.hir().body(eid);
                self.check_raw_ptr(cx, sig.header.unsafety, &sig.decl, body, item.id);
            }
        }
    }
}

impl<'a, 'tcx> Functions {
    fn check_arg_number(self, cx: &LateContext<'_, '_>, decl: &hir::FnDecl, span: Span) {
        let args = decl.inputs.len() as u64;
        if args > self.threshold {
            span_lint(
                cx,
                TOO_MANY_ARGUMENTS,
                span,
                &format!("this function has too many arguments ({}/{})", args, self.threshold),
            );
        }
    }

    fn check_raw_ptr(
        self,
        cx: &LateContext<'a, 'tcx>,
        unsafety: hir::Unsafety,
        decl: &'tcx hir::FnDecl,
        body: &'tcx hir::Body,
        nodeid: ast::NodeId,
    ) {
        let expr = &body.value;
        if unsafety == hir::Unsafety::Normal && cx.access_levels.is_exported(nodeid) {
            let raw_ptrs = iter_input_pats(decl, body)
                .zip(decl.inputs.iter())
                .filter_map(|(arg, ty)| raw_ptr_arg(arg, ty))
                .collect::<FxHashSet<_>>();

            if !raw_ptrs.is_empty() {
                let tables = cx.tcx.body_tables(body.id());
                let mut v = DerefVisitor {
                    cx,
                    ptrs: raw_ptrs,
                    tables,
                };

                hir::intravisit::walk_expr(&mut v, expr);
            }
        }
    }
}

fn raw_ptr_arg(arg: &hir::Arg, ty: &hir::Ty) -> Option<ast::NodeId> {
    if let (&hir::PatKind::Binding(_, id, _, _), &hir::TyKind::Ptr(_)) = (&arg.pat.node, &ty.node) {
        Some(id)
    } else {
        None
    }
}

struct DerefVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    ptrs: FxHashSet<ast::NodeId>,
    tables: &'a ty::TypeckTables<'tcx>,
}

impl<'a, 'tcx> hir::intravisit::Visitor<'tcx> for DerefVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        match expr.node {
            hir::ExprKind::Call(ref f, ref args) => {
                let ty = self.tables.expr_ty(f);

                if type_is_unsafe_function(self.cx, ty) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprKind::MethodCall(_, _, ref args) => {
                let def_id = self.tables.type_dependent_defs()[expr.hir_id].def_id();
                let base_type = self.cx.tcx.type_of(def_id);

                if type_is_unsafe_function(self.cx, base_type) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprKind::Unary(hir::UnDeref, ref ptr) => self.check_arg(ptr),
            _ => (),
        }

        hir::intravisit::walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
        intravisit::NestedVisitorMap::None
    }
}

impl<'a, 'tcx: 'a> DerefVisitor<'a, 'tcx> {
    fn check_arg(&self, ptr: &hir::Expr) {
        if let hir::ExprKind::Path(ref qpath) = ptr.node {
            if let Def::Local(id) = self.cx.tables.qpath_def(qpath, ptr.hir_id) {
                if self.ptrs.contains(&id) {
                    span_lint(
                        self.cx,
                        NOT_UNSAFE_PTR_ARG_DEREF,
                        ptr.span,
                        "this public function dereferences a raw pointer but is not marked `unsafe`",
                    );
                }
            }
        }
    }
}
