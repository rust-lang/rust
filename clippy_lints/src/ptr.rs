//! Checks for usage of  `&Vec[_]` and `&String`.

use rustc::hir::*;
use rustc::hir::map::NodeItem;
use rustc::lint::*;
use rustc::ty;
use syntax::ast::NodeId;
use syntax::codemap::Span;
use syntax_pos::MultiSpan;
use utils::{match_qpath, match_type, paths, span_lint, span_lint_and_then};

/// **What it does:** This lint checks for function arguments of type `&String`
/// or `&Vec` unless
/// the references are mutable.
///
/// **Why is this bad?** Requiring the argument to be of the specific size
/// makes the function less
/// useful for no benefit; slices in the form of `&[T]` or `&str` usually
/// suffice and can be
/// obtained from other types, too.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(&Vec<u32>) { .. }
/// ```
declare_lint! {
    pub PTR_ARG,
    Warn,
    "fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` \
     instead, respectively"
}

/// **What it does:** This lint checks for equality comparisons with `ptr::null`
///
/// **Why is this bad?** It's easier and more readable to use the inherent
/// `.is_null()`
/// method instead
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// if x == ptr::null { .. }
/// ```
declare_lint! {
    pub CMP_NULL,
    Warn,
    "comparing a pointer to a null pointer, suggesting to use `.is_null()` instead."
}

/// **What it does:** This lint checks for functions that take immutable
/// references and return
/// mutable ones.
///
/// **Why is this bad?** This is trivially unsound, as one can create two
/// mutable references
/// from the same (immutable!) source. This
/// [error](https://github.com/rust-lang/rust/issues/39465)
/// actually lead to an interim Rust release 1.15.1.
///
/// **Known problems:** To be on the conservative side, if there's at least one
/// mutable reference
/// with the output lifetime, this lint will not trigger. In practice, this
/// case is unlikely anyway.
///
/// **Example:**
/// ```rust
/// fn foo(&Foo) -> &mut Bar { .. }
/// ```
declare_lint! {
    pub MUT_FROM_REF,
    Warn,
    "fns that create mutable refs from immutable ref args"
}

#[derive(Copy, Clone)]
pub struct PointerPass;

impl LintPass for PointerPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(PTR_ARG, CMP_NULL, MUT_FROM_REF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PointerPass {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let ItemFn(ref decl, _, _, _, _, _) = item.node {
            check_fn(cx, decl, item.id);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx ImplItem) {
        if let ImplItemKind::Method(ref sig, _) = item.node {
            if let Some(NodeItem(it)) = cx.tcx.hir.find(cx.tcx.hir.get_parent(item.id)) {
                if let ItemImpl(_, _, _, _, Some(_), _, _) = it.node {
                    return; // ignore trait impls
                }
            }
            check_fn(cx, &sig.decl, item.id);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if let TraitItemKind::Method(ref sig, _) = item.node {
            check_fn(cx, &sig.decl, item.id);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprBinary(ref op, ref l, ref r) = expr.node {
            if (op.node == BiEq || op.node == BiNe) && (is_null_path(l) || is_null_path(r)) {
                span_lint(
                    cx,
                    CMP_NULL,
                    expr.span,
                    "Comparing with null is better expressed by the .is_null() method",
                );
            }
        }
    }
}

fn check_fn(cx: &LateContext, decl: &FnDecl, fn_id: NodeId) {
    let fn_def_id = cx.tcx.hir.local_def_id(fn_id);
    let sig = cx.tcx.fn_sig(fn_def_id);
    let fn_ty = sig.skip_binder();

    for (arg, ty) in decl.inputs.iter().zip(fn_ty.inputs()) {
        if let ty::TyRef(_,
                         ty::TypeAndMut {
                             ty,
                             mutbl: MutImmutable,
                         }) = ty.sty
        {
            if match_type(cx, ty, &paths::VEC) {
                span_lint(
                    cx,
                    PTR_ARG,
                    arg.span,
                    "writing `&Vec<_>` instead of `&[_]` involves one more reference and cannot be used \
                           with non-Vec-based slices. Consider changing the type to `&[...]`",
                );
            } else if match_type(cx, ty, &paths::STRING) {
                span_lint(
                    cx,
                    PTR_ARG,
                    arg.span,
                    "writing `&String` instead of `&str` involves a new object where a slice will do. \
                           Consider changing the type to `&str`",
                );
            }
        }
    }

    if let FunctionRetTy::Return(ref ty) = decl.output {
        if let Some((out, MutMutable, _)) = get_rptr_lm(ty) {
            let mut immutables = vec![];
            for (_, ref mutbl, ref argspan) in
                decl.inputs.iter().filter_map(|ty| get_rptr_lm(ty)).filter(
                    |&(lt, _, _)| lt.name == out.name,
                )
            {
                if *mutbl == MutMutable {
                    return;
                }
                immutables.push(*argspan);
            }
            if immutables.is_empty() {
                return;
            }
            span_lint_and_then(cx, MUT_FROM_REF, ty.span, "mutable borrow from immutable input(s)", |db| {
                let ms = MultiSpan::from_spans(immutables);
                db.span_note(ms, "immutable borrow here");
            });
        }
    }
}

fn get_rptr_lm(ty: &Ty) -> Option<(&Lifetime, Mutability, Span)> {
    if let Ty_::TyRptr(ref lt, ref m) = ty.node {
        Some((lt, m.mutbl, ty.span))
    } else {
        None
    }
}

fn is_null_path(expr: &Expr) -> bool {
    if let ExprCall(ref pathexp, ref args) = expr.node {
        if args.is_empty() {
            if let ExprPath(ref path) = pathexp.node {
                return match_qpath(path, &paths::PTR_NULL) || match_qpath(path, &paths::PTR_NULL_MUT);
            }
        }
    }
    false
}
