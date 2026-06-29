use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeQPath;
use clippy_utils::ty::{option_or_result_arg_ty, same_type_modulo_regions};
use clippy_utils::{is_from_proc_macro, last_path_segment, over};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Namespace, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{Body, Expr, ExprKind, PatKind, Safety};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_span::symbol::Ident;

use super::UNNECESSARY_UNWRAP_UNCHECKED;

#[derive(Clone, Copy, Debug)]
enum Variant {
    /// Free `fn` in a module
    Fn,
    /// Associated item from an `impl`
    Assoc(AssocKind),
}

impl Variant {
    fn msg(self) -> &'static str {
        // Don't use `format!` instead -- it won't be optimized out.
        match self {
            Variant::Fn => "usage of `unwrap_unchecked` when an `_unchecked` variant of the function exists",
            Variant::Assoc(AssocKind::Fn) => {
                "usage of `unwrap_unchecked` when an `_unchecked` variant of the associated function exists"
            },
            Variant::Assoc(AssocKind::Method) => {
                "usage of `unwrap_unchecked` when an `_unchecked` variant of the method exists"
            },
        }
    }
}

/// This only exists so the help message shows `associated function` or `method`, depending on
/// whether it has a `self` parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssocKind {
    /// No `self`: `fn new() -> Self`
    Fn,
    /// Has `self`: `fn ty<'tcx>(&self) -> Ty<'tcx>`
    Method,
}

impl AssocKind {
    fn new(fn_has_self_parameter: bool) -> Self {
        if fn_has_self_parameter { Self::Method } else { Self::Fn }
    }
}

fn unchecked_ident(checked_ident: Ident) -> Option<Ident> {
    let checked_ident = checked_ident.to_string();
    // Only add `_unchecked` if it doesn't already end with `_`
    (!checked_ident.ends_with('_')).then(|| Ident::from_str(&(checked_ident + "_unchecked")))
}

/// Find a function called the same as `checked`, but with added `_unchecked`.
///
/// This doesn't check if the methods are actually "similar" -- for that, see
/// [`same_functions_modulo_safety`]
fn find_unchecked_sibling_fn(
    cx: &LateContext<'_>,
    checked_def_id: DefId,
    checked_ident: Ident,
) -> Option<(DefId, Ident)> {
    // Don't use `parent_module`. We only want to lint if its first parent is a `Mod`,
    // i.e. if this is a free-standing function
    let parent = cx.tcx.parent(checked_def_id);
    if cx.tcx.def_kind(parent) == DefKind::Mod
        && let children = parent.as_local().map_or_else(
            || cx.tcx.module_children(parent),
            // We must use a !query for local modules to prevent an ICE.
            |parent| cx.tcx.module_children_local(parent),
        )
        // Make sure that there are other functions in this module
        // (otherwise there couldn't be an unchecked version)
        && children.len() > 1
        && let Some(unchecked_ident) = unchecked_ident(checked_ident)
        && let Some(unchecked_def_id) = children.iter().find_map(|child| {
            if child.ident == unchecked_ident
                && let Res::Def(DefKind::Fn, def_id) = child.res
            {
                Some(def_id)
            } else {
                None
            }
        })
    {
        Some((unchecked_def_id, unchecked_ident))
    } else {
        None
    }
}

/// Find a method called the same as `checked`, but with added `_unchecked`.
///
/// This doesn't check if the methods are actually "similar" -- for that, see
/// [`same_functions_modulo_safety`]
fn find_unchecked_sibling_method<'tcx>(
    cx: &LateContext<'tcx>,
    checked_def_id: DefId,
    checked_ident: Ident,
) -> Option<(&'tcx ty::AssocItem, Ident)> {
    // Don't use `parent_impl`. We only want to lint if its first parent is an `Impl`
    let parent = cx.tcx.parent(checked_def_id);
    if matches!(cx.tcx.def_kind(parent), DefKind::Impl { .. })
        && let Some(unchecked_ident) = unchecked_ident(checked_ident)
        // Only look in the same impl (to avoid dealing with generics etc.)
        && let Some(unchecked) = cx.tcx.associated_items(parent).find_by_ident_and_namespace(
            cx.tcx,
            unchecked_ident,
            Namespace::ValueNS,
            parent,
        )
    {
        Some((unchecked, unchecked_ident))
    } else {
        None
    }
}

/// Checks that `checked_def_id` and `unchecked_def_id` refer to functions with:
/// - same visibility
/// - identical signatures, apart from unsafety
/// - "matching" return types: the checked version returns `Option<T>`/`Result<T, E>`, while the
///   unchecked one returns `T`
fn same_functions_modulo_safety<'tcx>(
    cx: &LateContext<'tcx>,
    checked_def_id: DefId,
    unchecked_def_id: DefId,
    unwrapped_ret_ty: Ty<'tcx>,
) -> bool {
    let hir_body = |def_id: DefId| -> Option<&'tcx Body<'tcx>> { cx.tcx.hir_maybe_body_owned_by(def_id.as_local()?) };
    let fn_sig = |def_id| cx.tcx.fn_sig(def_id).skip_binder().skip_binder();

    if match (hir_body(checked_def_id), hir_body(unchecked_def_id)) {
        // For local functions, we can get the parameter names. In that case, we want to make sure
        // that the latter are equal between the checked and unchecked versions.
        (Some(checked_body), Some(unchecked_body)) => {
            over(checked_body.params, unchecked_body.params, |p1, p2| {
                // We only allow simple params (plain bindings) for now, to stay on the safer side.
                if let PatKind::Binding(bm1, _, ident1, None) = p1.pat.kind
                    && let PatKind::Binding(bm2, _, ident2, None) = p2.pat.kind
                {
                    bm1 == bm2 && ident1 == ident2
                } else {
                    false
                }
            })
        },
        // For non-local functions, parameter names are not accessible. Oh well, we'll let it slip
        (None, None) => true,
        // If only one of the versions is non-local, then something weird happened. Bail just in case
        _ => false,
    } {
        // Check that the functions have identical signatures, apart from safety, and return type (see
        // below)
        let checked_fn_sig = fn_sig(checked_def_id);
        let unchecked_fn_sig = fn_sig(unchecked_def_id);

        (checked_fn_sig.safety() == Safety::Safe && unchecked_fn_sig.safety() == Safety::Unsafe)
            && checked_fn_sig.c_variadic() == unchecked_fn_sig.c_variadic()
            && checked_fn_sig.abi() == unchecked_fn_sig.abi()
            // NOTE: the reason we use `same_type_modulo_regions` all over the place here is that
            // the regions of different functions will be distinct, even if they are called the same
            && over(checked_fn_sig.inputs(), unchecked_fn_sig.inputs(), |ty1, ty2| {
                same_type_modulo_regions(*ty1, *ty2)
            })
            // The checked version should return `Option<T>` or `Result<T, E>`,
            // and the unchecked version should return just `T`
            && same_type_modulo_regions(unchecked_fn_sig.output(), unwrapped_ret_ty)
            && option_or_result_arg_ty(cx, checked_fn_sig.output())
                .is_some_and(|wrapped_ty| same_type_modulo_regions(wrapped_ty, unwrapped_ret_ty))
            // Check that the visibilities are the same (for the purposes of replacing, it would be enough to have
            // the former _at least as_ visible as the latter, but we don't bother)
            && cx.tcx.visibility(unchecked_def_id) == cx.tcx.visibility(checked_def_id)
    } else {
        false
    }
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, recv: &Expr<'_>, call_span: Span) {
    if expr.span.from_expansion() {
        return;
    }
    let expected_ret_ty = cx.typeck_results().expr_ty(expr);
    let (variant, checked_span, unchecked_sugg, unchecked_full_path) = match recv.kind {
        // Construct `Variant::Fn(_)`, if applicable. This is necessary for us to handle
        // functions like `std::str::from_utf8_unchecked`.
        ExprKind::Call(path, _)
            if let ExprKind::Path(qpath) = path.kind
                && let checked_ident = last_path_segment(&qpath).ident
                && let checked_def_id = path.res(cx).def_id()
                && let Some((unchecked_def_id, unchecked_ident)) =
                    find_unchecked_sibling_fn(cx, checked_def_id, checked_ident)
                && same_functions_modulo_safety(cx, checked_def_id, unchecked_def_id, expected_ret_ty) =>
        {
            let unchecked_full_path = cx.tcx.def_path_str(unchecked_def_id);
            (
                Variant::Fn,
                checked_ident.span,
                if checked_ident.span == path.span {
                    // replacing `bar(x)` with `bar_unchecked(x)`
                    // `bar_unchecked` might not be in scope, so suggest the full path
                    unchecked_full_path.clone()
                } else {
                    // replacing `foo::bar(x)` with `foo::bar_unchecked(x)`
                    // since the path is qualified, we can just replace the final segment
                    unchecked_ident.to_string()
                },
                unchecked_full_path,
            )
        },
        // We unfortunately must handle `A::a(&a)` and `a.a()` separately, this handles the
        // former
        ExprKind::Call(path, _)
            if let ExprKind::Path(qpath) = path.kind
                && let checked_ident = last_path_segment(&qpath).ident
                && let checked_def_id = path.res(cx).def_id()
                && let Some((unchecked, unchecked_ident)) =
                    find_unchecked_sibling_method(cx, checked_def_id, checked_ident)
                && let ty::AssocKind::Fn { has_self, .. } = unchecked.kind
                && same_functions_modulo_safety(cx, checked_def_id, unchecked.def_id, expected_ret_ty) =>
        {
            let unchecked_full_path = cx.tcx.def_path_str(unchecked.def_id);
            (
                Variant::Assoc(AssocKind::new(has_self)),
                // since this is basically a method call, we only need to replace the method ident
                checked_ident.span,
                unchecked_ident.to_string(),
                unchecked_full_path,
            )
        },
        // ... And now the latter ^^
        ExprKind::MethodCall(segment, _, _, _)
            if let checked_ident = segment.ident
                && let Some(checked_def_id) = cx.typeck_results().type_dependent_def_id(recv.hir_id)
                && let Some((unchecked, unchecked_ident)) =
                    find_unchecked_sibling_method(cx, checked_def_id, checked_ident)
                && same_functions_modulo_safety(cx, checked_def_id, unchecked.def_id, expected_ret_ty) =>
        {
            let unchecked_full_path = cx.tcx.def_path_str(unchecked.def_id);
            (
                Variant::Assoc(AssocKind::Method),
                // since this is a method call, we only need to replace the method ident
                checked_ident.span,
                unchecked_ident.to_string(),
                unchecked_full_path,
            )
        },
        _ => return,
    };

    if !is_from_proc_macro(cx, expr) {
        span_lint_and_then(cx, UNNECESSARY_UNWRAP_UNCHECKED, expr.span, variant.msg(), |diag| {
            let sugg = vec![
                // replace the function with the unchecked version
                (checked_span, unchecked_sugg),
                // remove the call to `.unwrap_unchecked()`
                (call_span.with_lo(recv.span.hi()), String::new()),
            ];
            diag.multipart_suggestion(
                format!("use `{unchecked_full_path}` instead, and remove the call to `.unwrap_unchecked()`"),
                sugg,
                // TODO: make this `MachineApplicable` when the function comes from std/alloc/core
                // The reasoning is that, if the function comes from std/alloc/core, its checked and unchecked are
                // pretty likely to have their semantics match.
                Applicability::MaybeIncorrect,
            );
        });
    }
}
