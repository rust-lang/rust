//! Checks for usage of  `&Vec[_]` and `&String`.

use crate::utils::ptr::get_spans;
use crate::utils::{match_qpath, match_type, paths, snippet_opt, span_lint, span_lint_and_then, walk_ptrs_hir_ty};
use if_chain::if_chain;
use rustc::hir::QPath;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use std::borrow::Cow;
use syntax::source_map::Span;
use syntax_pos::MultiSpan;

declare_clippy_lint! {
    /// **What it does:** This lint checks for function arguments of type `&String`
    /// or `&Vec` unless the references are mutable. It will also suggest you
    /// replace `.clone()` calls with the appropriate `.to_owned()`/`to_string()`
    /// calls.
    ///
    /// **Why is this bad?** Requiring the argument to be of the specific size
    /// makes the function less useful for no benefit; slices in the form of `&[T]`
    /// or `&str` usually suffice and can be obtained from other types, too.
    ///
    /// **Known problems:** The lint does not follow data. So if you have an
    /// argument `x` and write `let y = x; y.clone()` the lint will not suggest
    /// changing that `.clone()` to `.to_owned()`.
    ///
    /// Other functions called from this function taking a `&String` or `&Vec`
    /// argument may also fail to compile if you change the argument. Applying
    /// this lint on them will fix the problem, but they may be in other crates.
    ///
    /// Also there may be `fn(&Vec)`-typed references pointing to your function.
    /// If you have them, you will get a compiler error after applying this lint's
    /// suggestions. You then have the choice to undo your changes or change the
    /// type of the reference.
    ///
    /// Note that if the function is part of your public interface, there may be
    /// other crates referencing it you may not be aware. Carefully deprecate the
    /// function before applying the lint suggestions in this case.
    ///
    /// **Example:**
    /// ```ignore
    /// fn foo(&Vec<u32>) { .. }
    /// ```
    pub PTR_ARG,
    style,
    "fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively"
}

declare_clippy_lint! {
    /// **What it does:** This lint checks for equality comparisons with `ptr::null`
    ///
    /// **Why is this bad?** It's easier and more readable to use the inherent
    /// `.is_null()`
    /// method instead
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// if x == ptr::null {
    ///     ..
    /// }
    /// ```
    pub CMP_NULL,
    style,
    "comparing a pointer to a null pointer, suggesting to use `.is_null()` instead."
}

declare_clippy_lint! {
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
    /// ```ignore
    /// fn foo(&Foo) -> &mut Bar { .. }
    /// ```
    pub MUT_FROM_REF,
    correctness,
    "fns that create mutable refs from immutable ref args"
}

declare_lint_pass!(Ptr => [PTR_ARG, CMP_NULL, MUT_FROM_REF]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Ptr {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let ItemKind::Fn(ref decl, _, _, body_id) = item.node {
            check_fn(cx, decl, item.hir_id, Some(body_id));
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx ImplItem) {
        if let ImplItemKind::Method(ref sig, body_id) = item.node {
            let parent_item = cx.tcx.hir().get_parent_item(item.hir_id);
            if let Some(Node::Item(it)) = cx.tcx.hir().find(parent_item) {
                if let ItemKind::Impl(_, _, _, _, Some(_), _, _) = it.node {
                    return; // ignore trait impls
                }
            }
            check_fn(cx, &sig.decl, item.hir_id, Some(body_id));
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if let TraitItemKind::Method(ref sig, ref trait_method) = item.node {
            let body_id = if let TraitMethod::Provided(b) = *trait_method {
                Some(b)
            } else {
                None
            };
            check_fn(cx, &sig.decl, item.hir_id, body_id);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Binary(ref op, ref l, ref r) = expr.node {
            if (op.node == BinOpKind::Eq || op.node == BinOpKind::Ne) && (is_null_path(l) || is_null_path(r)) {
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

#[allow(clippy::too_many_lines)]
fn check_fn(cx: &LateContext<'_, '_>, decl: &FnDecl, fn_id: HirId, opt_body_id: Option<BodyId>) {
    let fn_def_id = cx.tcx.hir().local_def_id_from_hir_id(fn_id);
    let sig = cx.tcx.fn_sig(fn_def_id);
    let fn_ty = sig.skip_binder();

    for (idx, (arg, ty)) in decl.inputs.iter().zip(fn_ty.inputs()).enumerate() {
        if let ty::Ref(_, ty, MutImmutable) = ty.sty {
            if match_type(cx, ty, &paths::VEC) {
                let mut ty_snippet = None;
                if_chain! {
                    if let TyKind::Path(QPath::Resolved(_, ref path)) = walk_ptrs_hir_ty(arg).node;
                    if let Some(&PathSegment{args: Some(ref parameters), ..}) = path.segments.last();
                    then {
                        let types: Vec<_> = parameters.args.iter().filter_map(|arg| match arg {
                            GenericArg::Type(ty) => Some(ty),
                            _ => None,
                        }).collect();
                        if types.len() == 1 {
                            ty_snippet = snippet_opt(cx, types[0].span);
                        }
                    }
                };
                if let Some(spans) = get_spans(cx, opt_body_id, idx, &[("clone", ".to_owned()")]) {
                    span_lint_and_then(
                        cx,
                        PTR_ARG,
                        arg.span,
                        "writing `&Vec<_>` instead of `&[_]` involves one more reference and cannot be used \
                         with non-Vec-based slices.",
                        |db| {
                            if let Some(ref snippet) = ty_snippet {
                                db.span_suggestion(
                                    arg.span,
                                    "change this to",
                                    format!("&[{}]", snippet),
                                    Applicability::Unspecified,
                                );
                            }
                            for (clonespan, suggestion) in spans {
                                db.span_suggestion(
                                    clonespan,
                                    &snippet_opt(cx, clonespan).map_or("change the call to".into(), |x| {
                                        Cow::Owned(format!("change `{}` to", x))
                                    }),
                                    suggestion.into(),
                                    Applicability::Unspecified,
                                );
                            }
                        },
                    );
                }
            } else if match_type(cx, ty, &paths::STRING) {
                if let Some(spans) = get_spans(cx, opt_body_id, idx, &[("clone", ".to_string()"), ("as_str", "")]) {
                    span_lint_and_then(
                        cx,
                        PTR_ARG,
                        arg.span,
                        "writing `&String` instead of `&str` involves a new object where a slice will do.",
                        |db| {
                            db.span_suggestion(arg.span, "change this to", "&str".into(), Applicability::Unspecified);
                            for (clonespan, suggestion) in spans {
                                db.span_suggestion_short(
                                    clonespan,
                                    &snippet_opt(cx, clonespan).map_or("change the call to".into(), |x| {
                                        Cow::Owned(format!("change `{}` to", x))
                                    }),
                                    suggestion.into(),
                                    Applicability::Unspecified,
                                );
                            }
                        },
                    );
                }
            } else if match_type(cx, ty, &paths::COW) {
                if_chain! {
                    if let TyKind::Rptr(_, MutTy { ref ty, ..} ) = arg.node;
                    if let TyKind::Path(ref path) = ty.node;
                    if let QPath::Resolved(None, ref pp) = *path;
                    if let [ref bx] = *pp.segments;
                    if let Some(ref params) = bx.args;
                    if !params.parenthesized;
                    if let Some(inner) = params.args.iter().find_map(|arg| match arg {
                        GenericArg::Type(ty) => Some(ty),
                        _ => None,
                    });
                    then {
                        let replacement = snippet_opt(cx, inner.span);
                        if let Some(r) = replacement {
                            span_lint_and_then(
                                cx,
                                PTR_ARG,
                                arg.span,
                                "using a reference to `Cow` is not recommended.",
                                |db| {
                                    db.span_suggestion(
                                        arg.span,
                                        "change this to",
                                        "&".to_owned() + &r,
                                        Applicability::Unspecified,
                                    );
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    if let FunctionRetTy::Return(ref ty) = decl.output {
        if let Some((out, MutMutable, _)) = get_rptr_lm(ty) {
            let mut immutables = vec![];
            for (_, ref mutbl, ref argspan) in decl
                .inputs
                .iter()
                .filter_map(|ty| get_rptr_lm(ty))
                .filter(|&(lt, _, _)| lt.name == out.name)
            {
                if *mutbl == MutMutable {
                    return;
                }
                immutables.push(*argspan);
            }
            if immutables.is_empty() {
                return;
            }
            span_lint_and_then(
                cx,
                MUT_FROM_REF,
                ty.span,
                "mutable borrow from immutable input(s)",
                |db| {
                    let ms = MultiSpan::from_spans(immutables);
                    db.span_note(ms, "immutable borrow here");
                },
            );
        }
    }
}

fn get_rptr_lm(ty: &Ty) -> Option<(&Lifetime, Mutability, Span)> {
    if let TyKind::Rptr(ref lt, ref m) = ty.node {
        Some((lt, m.mutbl, ty.span))
    } else {
        None
    }
}

fn is_null_path(expr: &Expr) -> bool {
    if let ExprKind::Call(ref pathexp, ref args) = expr.node {
        if args.is_empty() {
            if let ExprKind::Path(ref path) = pathexp.node {
                return match_qpath(path, &paths::PTR_NULL) || match_qpath(path, &paths::PTR_NULL_MUT);
            }
        }
    }
    false
}
