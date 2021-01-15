//! Checks for usage of  `&Vec[_]` and `&String`.

use crate::utils::ptr::get_spans;
use crate::utils::{
    is_allowed, is_type_diagnostic_item, match_qpath, match_type, paths, snippet_opt, span_lint, span_lint_and_sugg,
    span_lint_and_then, walk_ptrs_hir_ty,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{
    BinOpKind, BodyId, Expr, ExprKind, FnDecl, FnRetTy, GenericArg, HirId, Impl, ImplItem, ImplItemKind, Item,
    ItemKind, Lifetime, MutTy, Mutability, Node, PathSegment, QPath, TraitFn, TraitItem, TraitItemKind, Ty, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::{sym, MultiSpan};
use std::borrow::Cow;

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
    /// One notable example of a function that may cause issues, and which cannot
    /// easily be changed due to being in the standard library is `Vec::contains`.
    /// when called on a `Vec<Vec<T>>`. If a `&Vec` is passed to that method then
    /// it will compile, but if a `&[T]` is passed then it will not compile.
    ///
    /// ```ignore
    /// fn cannot_take_a_slice(v: &Vec<u8>) -> bool {
    ///     let vec_of_vecs: Vec<Vec<u8>> = some_other_fn();
    ///
    ///     vec_of_vecs.contains(v)
    /// }
    /// ```
    ///
    /// Also there may be `fn(&Vec)`-typed references pointing to your function.
    /// If you have them, you will get a compiler error after applying this lint's
    /// suggestions. You then have the choice to undo your changes or change the
    /// type of the reference.
    ///
    /// Note that if the function is part of your public interface, there may be
    /// other crates referencing it, of which you may not be aware. Carefully
    /// deprecate the function before applying the lint suggestions in this case.
    ///
    /// **Example:**
    /// ```ignore
    /// // Bad
    /// fn foo(&Vec<u32>) { .. }
    ///
    /// // Good
    /// fn foo(&[u32]) { .. }
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
    /// // Bad
    /// if x == ptr::null {
    ///     ..
    /// }
    ///
    /// // Good
    /// if x.is_null() {
    ///     ..
    /// }
    /// ```
    pub CMP_NULL,
    style,
    "comparing a pointer to a null pointer, suggesting to use `.is_null()` instead."
}

declare_clippy_lint! {
    /// **What it does:** This lint checks for functions that take immutable
    /// references and return mutable ones.
    ///
    /// **Why is this bad?** This is trivially unsound, as one can create two
    /// mutable references from the same (immutable!) source.
    /// This [error](https://github.com/rust-lang/rust/issues/39465)
    /// actually lead to an interim Rust release 1.15.1.
    ///
    /// **Known problems:** To be on the conservative side, if there's at least one
    /// mutable reference with the output lifetime, this lint will not trigger.
    /// In practice, this case is unlikely anyway.
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

impl<'tcx> LateLintPass<'tcx> for Ptr {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Fn(ref sig, _, body_id) = item.kind {
            check_fn(cx, &sig.decl, item.hir_id, Some(body_id));
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(ref sig, body_id) = item.kind {
            let parent_item = cx.tcx.hir().get_parent_item(item.hir_id);
            if let Some(Node::Item(it)) = cx.tcx.hir().find(parent_item) {
                if let ItemKind::Impl(Impl { of_trait: Some(_), .. }) = it.kind {
                    return; // ignore trait impls
                }
            }
            check_fn(cx, &sig.decl, item.hir_id, Some(body_id));
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(ref sig, ref trait_method) = item.kind {
            let body_id = if let TraitFn::Provided(b) = *trait_method {
                Some(b)
            } else {
                None
            };
            check_fn(cx, &sig.decl, item.hir_id, body_id);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, ref l, ref r) = expr.kind {
            if (op.node == BinOpKind::Eq || op.node == BinOpKind::Ne) && (is_null_path(l) || is_null_path(r)) {
                span_lint(
                    cx,
                    CMP_NULL,
                    expr.span,
                    "comparing with null is better expressed by the `.is_null()` method",
                );
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
fn check_fn(cx: &LateContext<'_>, decl: &FnDecl<'_>, fn_id: HirId, opt_body_id: Option<BodyId>) {
    let fn_def_id = cx.tcx.hir().local_def_id(fn_id);
    let sig = cx.tcx.fn_sig(fn_def_id);
    let fn_ty = sig.skip_binder();
    let body = opt_body_id.map(|id| cx.tcx.hir().body(id));

    for (idx, (arg, ty)) in decl.inputs.iter().zip(fn_ty.inputs()).enumerate() {
        // Honor the allow attribute on parameters. See issue 5644.
        if let Some(body) = &body {
            if is_allowed(cx, PTR_ARG, body.params[idx].hir_id) {
                continue;
            }
        }

        if let ty::Ref(_, ty, Mutability::Not) = ty.kind() {
            if is_type_diagnostic_item(cx, ty, sym::vec_type) {
                if let Some(spans) = get_spans(cx, opt_body_id, idx, &[("clone", ".to_owned()")]) {
                    span_lint_and_then(
                        cx,
                        PTR_ARG,
                        arg.span,
                        "writing `&Vec<_>` instead of `&[_]` involves one more reference and cannot be used \
                         with non-Vec-based slices.",
                        |diag| {
                            if let Some(ref snippet) = get_only_generic_arg_snippet(cx, arg) {
                                diag.span_suggestion(
                                    arg.span,
                                    "change this to",
                                    format!("&[{}]", snippet),
                                    Applicability::Unspecified,
                                );
                            }
                            for (clonespan, suggestion) in spans {
                                diag.span_suggestion(
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
            } else if is_type_diagnostic_item(cx, ty, sym::string_type) {
                if let Some(spans) = get_spans(cx, opt_body_id, idx, &[("clone", ".to_string()"), ("as_str", "")]) {
                    span_lint_and_then(
                        cx,
                        PTR_ARG,
                        arg.span,
                        "writing `&String` instead of `&str` involves a new object where a slice will do.",
                        |diag| {
                            diag.span_suggestion(arg.span, "change this to", "&str".into(), Applicability::Unspecified);
                            for (clonespan, suggestion) in spans {
                                diag.span_suggestion_short(
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
            } else if match_type(cx, ty, &paths::PATH_BUF) {
                if let Some(spans) = get_spans(cx, opt_body_id, idx, &[("clone", ".to_path_buf()"), ("as_path", "")]) {
                    span_lint_and_then(
                        cx,
                        PTR_ARG,
                        arg.span,
                        "writing `&PathBuf` instead of `&Path` involves a new object where a slice will do.",
                        |diag| {
                            diag.span_suggestion(
                                arg.span,
                                "change this to",
                                "&Path".into(),
                                Applicability::Unspecified,
                            );
                            for (clonespan, suggestion) in spans {
                                diag.span_suggestion_short(
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
                    if let TyKind::Rptr(_, MutTy { ref ty, ..} ) = arg.kind;
                    if let TyKind::Path(ref path) = ty.kind;
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
                            span_lint_and_sugg(
                                cx,
                                PTR_ARG,
                                arg.span,
                                "using a reference to `Cow` is not recommended.",
                                "change this to",
                                "&".to_owned() + &r,
                                Applicability::Unspecified,
                            );
                        }
                    }
                }
            }
        }
    }

    if let FnRetTy::Return(ref ty) = decl.output {
        if let Some((out, Mutability::Mut, _)) = get_rptr_lm(ty) {
            let mut immutables = vec![];
            for (_, ref mutbl, ref argspan) in decl
                .inputs
                .iter()
                .filter_map(|ty| get_rptr_lm(ty))
                .filter(|&(lt, _, _)| lt.name == out.name)
            {
                if *mutbl == Mutability::Mut {
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
                |diag| {
                    let ms = MultiSpan::from_spans(immutables);
                    diag.span_note(ms, "immutable borrow here");
                },
            );
        }
    }
}

fn get_only_generic_arg_snippet(cx: &LateContext<'_>, arg: &Ty<'_>) -> Option<String> {
    if_chain! {
        if let TyKind::Path(QPath::Resolved(_, ref path)) = walk_ptrs_hir_ty(arg).kind;
        if let Some(&PathSegment{args: Some(ref parameters), ..}) = path.segments.last();
        let types: Vec<_> = parameters.args.iter().filter_map(|arg| match arg {
            GenericArg::Type(ty) => Some(ty),
            _ => None,
        }).collect();
        if types.len() == 1;
        then {
            snippet_opt(cx, types[0].span)
        } else {
            None
        }
    }
}

fn get_rptr_lm<'tcx>(ty: &'tcx Ty<'tcx>) -> Option<(&'tcx Lifetime, Mutability, Span)> {
    if let TyKind::Rptr(ref lt, ref m) = ty.kind {
        Some((lt, m.mutbl, ty.span))
    } else {
        None
    }
}

fn is_null_path(expr: &Expr<'_>) -> bool {
    if let ExprKind::Call(ref pathexp, ref args) = expr.kind {
        if args.is_empty() {
            if let ExprKind::Path(ref path) = pathexp.kind {
                return match_qpath(path, &paths::PTR_NULL) || match_qpath(path, &paths::PTR_NULL_MUT);
            }
        }
    }
    false
}
