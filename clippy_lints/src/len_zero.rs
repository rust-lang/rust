use rustc::lint::*;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::hir::*;
use std::collections::HashSet;
use syntax::ast::{Lit, LitKind, Name};
use syntax::codemap::{Span, Spanned};
use utils::{get_item_name, in_macro, snippet, span_lint, span_lint_and_sugg, walk_ptrs_ty};

/// **What it does:** Checks for getting the length of something via `.len()`
/// just to compare to zero, and suggests using `.is_empty()` where applicable.
///
/// **Why is this bad?** Some structures can answer `.is_empty()` much faster
/// than calculating their length. Notably, for slices, getting the length
/// requires a subtraction whereas `.is_empty()` is just a comparison. So it is
/// good to get into the habit of using `.is_empty()`, and having it is cheap.
/// Besides, it makes the intent clearer than a manual comparison.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// if x.len() == 0 { .. }
/// ```
declare_lint! {
    pub LEN_ZERO,
    Warn,
    "checking `.len() == 0` or `.len() > 0` (or similar) when `.is_empty()` \
     could be used instead"
}

/// **What it does:** Checks for items that implement `.len()` but not
/// `.is_empty()`.
///
/// **Why is this bad?** It is good custom to have both methods, because for
/// some data structures, asking about the length will be a costly operation,
/// whereas `.is_empty()` can usually answer in constant time. Also it used to
/// lead to false positives on the [`len_zero`](#len_zero) lint – currently that
/// lint will ignore such entities.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// impl X {
///     pub fn len(&self) -> usize { .. }
/// }
/// ```
declare_lint! {
    pub LEN_WITHOUT_IS_EMPTY,
    Warn,
    "traits or impls with a public `len` method but no corresponding `is_empty` method"
}

#[derive(Copy, Clone)]
pub struct LenZero;

impl LintPass for LenZero {
    fn get_lints(&self) -> LintArray {
        lint_array!(LEN_ZERO, LEN_WITHOUT_IS_EMPTY)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LenZero {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if in_macro(item.span) {
            return;
        }

        match item.node {
            ItemTrait(_, _, _, ref trait_items) => check_trait_items(cx, item, trait_items),
            ItemImpl(_, _, _, _, None, _, ref impl_items) => check_impl_items(cx, item, impl_items),
            _ => (),
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if in_macro(expr.span) {
            return;
        }

        if let ExprBinary(Spanned { node: cmp, .. }, ref left, ref right) = expr.node {
            match cmp {
                BiEq => check_cmp(cx, expr.span, left, right, ""),
                BiGt | BiNe => check_cmp(cx, expr.span, left, right, "!"),
                _ => (),
            }
        }
    }
}

fn check_trait_items(cx: &LateContext, visited_trait: &Item, trait_items: &[TraitItemRef]) {
    fn is_named_self(cx: &LateContext, item: &TraitItemRef, name: &str) -> bool {
        item.name == name && if let AssociatedItemKind::Method { has_self } = item.kind {
            has_self && {
                let did = cx.tcx.hir.local_def_id(item.id.node_id);
                cx.tcx.fn_sig(did).inputs().skip_binder().len() == 1
            }
        } else {
            false
        }
    }

    // fill the set with current and super traits
    fn fill_trait_set(traitt: DefId, set: &mut HashSet<DefId>, cx: &LateContext) {
        if set.insert(traitt) {
            for supertrait in ::rustc::traits::supertrait_def_ids(cx.tcx, traitt) {
                fill_trait_set(supertrait, set, cx);
            }
        }
    }

    if cx.access_levels.is_exported(visited_trait.id) && trait_items.iter().any(|i| is_named_self(cx, i, "len")) {
        let mut current_and_super_traits = HashSet::new();
        let visited_trait_def_id = cx.tcx.hir.local_def_id(visited_trait.id);
        fill_trait_set(visited_trait_def_id, &mut current_and_super_traits, cx);

        let is_empty_method_found = current_and_super_traits
            .iter()
            .flat_map(|&i| cx.tcx.associated_items(i))
            .any(|i| {
                i.kind == ty::AssociatedKind::Method && i.method_has_self_argument && i.name == "is_empty" &&
                    cx.tcx.fn_sig(i.def_id).inputs().skip_binder().len() == 1
            });

        if !is_empty_method_found {
            span_lint(
                cx,
                LEN_WITHOUT_IS_EMPTY,
                visited_trait.span,
                &format!(
                    "trait `{}` has a `len` method but no (possibly inherited) `is_empty` method",
                    visited_trait.name
                ),
            );
        }
    }
}

fn check_impl_items(cx: &LateContext, item: &Item, impl_items: &[ImplItemRef]) {
    fn is_named_self(cx: &LateContext, item: &ImplItemRef, name: &str) -> bool {
        item.name == name && if let AssociatedItemKind::Method { has_self } = item.kind {
            has_self && {
                let did = cx.tcx.hir.local_def_id(item.id.node_id);
                cx.tcx.fn_sig(did).inputs().skip_binder().len() == 1
            }
        } else {
            false
        }
    }

    let is_empty = if let Some(is_empty) = impl_items.iter().find(|i| is_named_self(cx, i, "is_empty")) {
        if cx.access_levels.is_exported(is_empty.id.node_id) {
            return;
        } else {
            "a private"
        }
    } else {
        "no corresponding"
    };

    if let Some(i) = impl_items.iter().find(|i| is_named_self(cx, i, "len")) {
        if cx.access_levels.is_exported(i.id.node_id) {
            let def_id = cx.tcx.hir.local_def_id(item.id);
            let ty = cx.tcx.type_of(def_id);

            span_lint(
                cx,
                LEN_WITHOUT_IS_EMPTY,
                item.span,
                &format!("item `{}` has a public `len` method but {} `is_empty` method", ty, is_empty),
            );
        }
    }
}

fn check_cmp(cx: &LateContext, span: Span, left: &Expr, right: &Expr, op: &str) {
    // check if we are in an is_empty() method
    if let Some(name) = get_item_name(cx, left) {
        if name == "is_empty" {
            return;
        }
    }
    match (&left.node, &right.node) {
        (&ExprLit(ref lit), &ExprMethodCall(ref method_path, _, ref args)) |
        (&ExprMethodCall(ref method_path, _, ref args), &ExprLit(ref lit)) => {
            check_len_zero(cx, span, method_path.name, args, lit, op)
        },
        _ => (),
    }
}

fn check_len_zero(cx: &LateContext, span: Span, name: Name, args: &[Expr], lit: &Lit, op: &str) {
    if let Spanned {
        node: LitKind::Int(0, _),
        ..
    } = *lit
    {
        if name == "len" && args.len() == 1 && has_is_empty(cx, &args[0]) {
            span_lint_and_sugg(
                cx,
                LEN_ZERO,
                span,
                "length comparison to zero",
                "using `is_empty` is more concise",
                format!("{}{}.is_empty()", op, snippet(cx, args[0].span, "_")),
            );
        }
    }
}

/// Check if this type has an `is_empty` method.
fn has_is_empty(cx: &LateContext, expr: &Expr) -> bool {
    /// Get an `AssociatedItem` and return true if it matches `is_empty(self)`.
    fn is_is_empty(cx: &LateContext, item: &ty::AssociatedItem) -> bool {
        if let ty::AssociatedKind::Method = item.kind {
            if item.name == "is_empty" {
                let sig = cx.tcx.fn_sig(item.def_id);
                let ty = sig.skip_binder();
                ty.inputs().len() == 1
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Check the inherent impl's items for an `is_empty(self)` method.
    fn has_is_empty_impl(cx: &LateContext, id: DefId) -> bool {
        cx.tcx.inherent_impls(id).iter().any(|imp| {
            cx.tcx
                .associated_items(*imp)
                .any(|item| is_is_empty(cx, &item))
        })
    }

    let ty = &walk_ptrs_ty(cx.tables.expr_ty(expr));
    match ty.sty {
        ty::TyDynamic(..) => cx.tcx
            .associated_items(ty.ty_to_def_id().expect("trait impl not found"))
            .any(|item| is_is_empty(cx, &item)),
        ty::TyProjection(_) => ty.ty_to_def_id()
            .map_or(false, |id| has_is_empty_impl(cx, id)),
        ty::TyAdt(id, _) => has_is_empty_impl(cx, id.did),
        ty::TyArray(..) | ty::TySlice(..) | ty::TyStr => true,
        _ => false,
    }
}
