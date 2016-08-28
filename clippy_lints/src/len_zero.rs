use rustc::lint::*;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, MethodTraitItemId, ImplOrTraitItemId};
use rustc::hir::*;
use syntax::ast::{Lit, LitKind, Name};
use syntax::codemap::{Span, Spanned};
use syntax::ptr::P;
use utils::{get_item_name, in_macro, snippet, span_lint, span_lint_and_then, walk_ptrs_ty};

/// **What it does:** Checks for getting the length of something via `.len()`
/// just to compare to zero, and suggests using `.is_empty()` where applicable.
///
/// **Why is this bad?** Some structures can answer `.is_empty()` much faster
/// than calculating their length. So it is good to get into the habit of using
/// `.is_empty()`, and having it is cheap. Besides, it makes the intent clearer
/// than a comparison.
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
///     fn len(&self) -> usize { .. }
/// }
/// ```
declare_lint! {
    pub LEN_WITHOUT_IS_EMPTY,
    Warn,
    "traits and impls that have `.len()` but not `.is_empty()`"
}

#[derive(Copy,Clone)]
pub struct LenZero;

impl LintPass for LenZero {
    fn get_lints(&self) -> LintArray {
        lint_array!(LEN_ZERO, LEN_WITHOUT_IS_EMPTY)
    }
}

impl LateLintPass for LenZero {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if in_macro(cx, item.span) {
            return;
        }

        match item.node {
            ItemTrait(_, _, _, ref trait_items) => check_trait_items(cx, item, trait_items),
            ItemImpl(_, _, _, None, _, ref impl_items) => check_impl_items(cx, item, impl_items),
            _ => (),
        }
    }

    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if in_macro(cx, expr.span) {
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

fn check_trait_items(cx: &LateContext, item: &Item, trait_items: &[TraitItem]) {
    fn is_named_self(item: &TraitItem, name: &str) -> bool {
        item.name.as_str() == name &&
        if let MethodTraitItem(ref sig, _) = item.node {
            is_self_sig(sig)
        } else {
            false
        }
    }

    if !trait_items.iter().any(|i| is_named_self(i, "is_empty")) {
        for i in trait_items {
            if is_named_self(i, "len") {
                span_lint(cx,
                          LEN_WITHOUT_IS_EMPTY,
                          i.span,
                          &format!("trait `{}` has a `.len(_: &Self)` method, but no `.is_empty(_: &Self)` method. \
                                    Consider adding one",
                                   item.name));
            }
        }
    }
}

fn check_impl_items(cx: &LateContext, item: &Item, impl_items: &[ImplItem]) {
    fn is_named_self(item: &ImplItem, name: &str) -> bool {
        item.name.as_str() == name &&
        if let ImplItemKind::Method(ref sig, _) = item.node {
            is_self_sig(sig)
        } else {
            false
        }
    }

    if !impl_items.iter().any(|i| is_named_self(i, "is_empty")) {
        for i in impl_items {
            if is_named_self(i, "len") {
                let ty = cx.tcx.node_id_to_type(item.id);

                span_lint(cx,
                          LEN_WITHOUT_IS_EMPTY,
                          i.span,
                          &format!("item `{}` has a `.len(_: &Self)` method, but no `.is_empty(_: &Self)` method. \
                                    Consider adding one",
                                   ty));
                return;
            }
        }
    }
}

fn is_self_sig(sig: &MethodSig) -> bool {
    if sig.decl.has_self() {
        sig.decl.inputs.len() == 1
    } else {
        false
    }
}

fn check_cmp(cx: &LateContext, span: Span, left: &Expr, right: &Expr, op: &str) {
    // check if we are in an is_empty() method
    if let Some(name) = get_item_name(cx, left) {
        if name.as_str() == "is_empty" {
            return;
        }
    }
    match (&left.node, &right.node) {
        (&ExprLit(ref lit), &ExprMethodCall(ref method, _, ref args)) |
        (&ExprMethodCall(ref method, _, ref args), &ExprLit(ref lit)) => {
            check_len_zero(cx, span, &method.node, args, lit, op)
        }
        _ => (),
    }
}

fn check_len_zero(cx: &LateContext, span: Span, name: &Name, args: &[P<Expr>], lit: &Lit, op: &str) {
    if let Spanned { node: LitKind::Int(0, _), .. } = *lit {
        if name.as_str() == "len" && args.len() == 1 && has_is_empty(cx, &args[0]) {
            span_lint_and_then(cx, LEN_ZERO, span, "length comparison to zero", |db| {
                db.span_suggestion(span,
                                   "consider using `is_empty`",
                                   format!("{}{}.is_empty()", op, snippet(cx, args[0].span, "_")));
            });
        }
    }
}

/// Check if this type has an `is_empty` method.
fn has_is_empty(cx: &LateContext, expr: &Expr) -> bool {
    /// Get an `ImplOrTraitItem` and return true if it matches `is_empty(self)`.
    fn is_is_empty(cx: &LateContext, id: &ImplOrTraitItemId) -> bool {
        if let MethodTraitItemId(def_id) = *id {
            if let ty::MethodTraitItem(ref method) = cx.tcx.impl_or_trait_item(def_id) {
                method.name.as_str() == "is_empty" && method.fty.sig.skip_binder().inputs.len() == 1
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Check the inherent impl's items for an `is_empty(self)` method.
    fn has_is_empty_impl(cx: &LateContext, id: &DefId) -> bool {
        let impl_items = cx.tcx.impl_items.borrow();
        cx.tcx.inherent_impls.borrow().get(id).map_or(false, |ids| {
            ids.iter().any(|iid| impl_items.get(iid).map_or(false, |iids| iids.iter().any(|i| is_is_empty(cx, i))))
        })
    }

    let ty = &walk_ptrs_ty(cx.tcx.expr_ty(expr));
    match ty.sty {
        ty::TyTrait(_) => {
            cx.tcx
              .trait_item_def_ids
              .borrow()
              .get(&ty.ty_to_def_id().expect("trait impl not found"))
              .map_or(false, |ids| ids.iter().any(|i| is_is_empty(cx, i)))
        }
        ty::TyProjection(_) => ty.ty_to_def_id().map_or(false, |id| has_is_empty_impl(cx, &id)),
        ty::TyEnum(id, _) |
        ty::TyStruct(id, _) |
        ty::TyUnion(id, _) => has_is_empty_impl(cx, &id.did),
        ty::TyArray(..) | ty::TyStr => true,
        _ => false,
    }
}
