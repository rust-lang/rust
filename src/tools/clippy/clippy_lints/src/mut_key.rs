use crate::utils::{match_def_path, paths, span_lint, trait_ref_of_method};
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{Adt, Array, RawPtr, Ref, Slice, Tuple, Ty, TypeAndMut};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for sets/maps with mutable key types.
    ///
    /// **Why is this bad?** All of `HashMap`, `HashSet`, `BTreeMap` and
    /// `BtreeSet` rely on either the hash or the order of keys be unchanging,
    /// so having types with interior mutability is a bad idea.
    ///
    /// **Known problems:** It's correct to use a struct, that contains interior mutability
    /// as a key, when its `Hash` implementation doesn't access any of the interior mutable types.
    /// However, this lint is unable to recognize this, so it causes a false positive in theses cases.
    /// The `bytes` crate is a great example of this.
    ///
    /// **Example:**
    /// ```rust
    /// use std::cmp::{PartialEq, Eq};
    /// use std::collections::HashSet;
    /// use std::hash::{Hash, Hasher};
    /// use std::sync::atomic::AtomicUsize;
    ///# #[allow(unused)]
    ///
    /// struct Bad(AtomicUsize);
    /// impl PartialEq for Bad {
    ///     fn eq(&self, rhs: &Self) -> bool {
    ///          ..
    /// ; unimplemented!();
    ///     }
    /// }
    ///
    /// impl Eq for Bad {}
    ///
    /// impl Hash for Bad {
    ///     fn hash<H: Hasher>(&self, h: &mut H) {
    ///         ..
    /// ; unimplemented!();
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let _: HashSet<Bad> = HashSet::new();
    /// }
    /// ```
    pub MUTABLE_KEY_TYPE,
    correctness,
    "Check for mutable `Map`/`Set` key type"
}

declare_lint_pass!(MutableKeyType => [ MUTABLE_KEY_TYPE ]);

impl<'tcx> LateLintPass<'tcx> for MutableKeyType {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        if let hir::ItemKind::Fn(ref sig, ..) = item.kind {
            check_sig(cx, item.hir_id, &sig.decl);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'tcx>) {
        if let hir::ImplItemKind::Fn(ref sig, ..) = item.kind {
            if trait_ref_of_method(cx, item.hir_id).is_none() {
                check_sig(cx, item.hir_id, &sig.decl);
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Fn(ref sig, ..) = item.kind {
            check_sig(cx, item.hir_id, &sig.decl);
        }
    }

    fn check_local(&mut self, cx: &LateContext<'_>, local: &hir::Local<'_>) {
        if let hir::PatKind::Wild = local.pat.kind {
            return;
        }
        check_ty(cx, local.span, cx.typeck_results().pat_ty(&*local.pat));
    }
}

fn check_sig<'tcx>(cx: &LateContext<'tcx>, item_hir_id: hir::HirId, decl: &hir::FnDecl<'_>) {
    let fn_def_id = cx.tcx.hir().local_def_id(item_hir_id);
    let fn_sig = cx.tcx.fn_sig(fn_def_id);
    for (hir_ty, ty) in decl.inputs.iter().zip(fn_sig.inputs().skip_binder().iter()) {
        check_ty(cx, hir_ty.span, ty);
    }
    check_ty(
        cx,
        decl.output.span(),
        cx.tcx.erase_late_bound_regions(fn_sig.output()),
    );
}

// We want to lint 1. sets or maps with 2. not immutable key types and 3. no unerased
// generics (because the compiler cannot ensure immutability for unknown types).
fn check_ty<'tcx>(cx: &LateContext<'tcx>, span: Span, ty: Ty<'tcx>) {
    let ty = ty.peel_refs();
    if let Adt(def, substs) = ty.kind() {
        if [&paths::HASHMAP, &paths::BTREEMAP, &paths::HASHSET, &paths::BTREESET]
            .iter()
            .any(|path| match_def_path(cx, def.did, &**path))
            && is_mutable_type(cx, substs.type_at(0), span)
        {
            span_lint(cx, MUTABLE_KEY_TYPE, span, "mutable key type");
        }
    }
}

fn is_mutable_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, span: Span) -> bool {
    match *ty.kind() {
        RawPtr(TypeAndMut { ty: inner_ty, mutbl }) | Ref(_, inner_ty, mutbl) => {
            mutbl == hir::Mutability::Mut || is_mutable_type(cx, inner_ty, span)
        },
        Slice(inner_ty) => is_mutable_type(cx, inner_ty, span),
        Array(inner_ty, size) => {
            size.try_eval_usize(cx.tcx, cx.param_env).map_or(true, |u| u != 0) && is_mutable_type(cx, inner_ty, span)
        },
        Tuple(..) => ty.tuple_fields().any(|ty| is_mutable_type(cx, ty, span)),
        Adt(..) => {
            cx.tcx.layout_of(cx.param_env.and(ty)).is_ok()
                && !ty.has_escaping_bound_vars()
                && !ty.is_freeze(cx.tcx.at(span), cx.param_env)
        },
        _ => false,
    }
}
