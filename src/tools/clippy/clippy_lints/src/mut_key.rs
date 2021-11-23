use clippy_utils::diagnostics::span_lint;
use clippy_utils::trait_ref_of_method;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TypeFoldable;
use rustc_middle::ty::{Adt, Array, Ref, Slice, Tuple, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for sets/maps with mutable key types.
    ///
    /// ### Why is this bad?
    /// All of `HashMap`, `HashSet`, `BTreeMap` and
    /// `BtreeSet` rely on either the hash or the order of keys be unchanging,
    /// so having types with interior mutability is a bad idea.
    ///
    /// ### Known problems
    ///
    /// #### False Positives
    /// It's correct to use a struct that contains interior mutability as a key, when its
    /// implementation of `Hash` or `Ord` doesn't access any of the interior mutable types.
    /// However, this lint is unable to recognize this, so it will often cause false positives in
    /// theses cases.  The `bytes` crate is a great example of this.
    ///
    /// #### False Negatives
    /// For custom `struct`s/`enum`s, this lint is unable to check for interior mutability behind
    /// indirection.  For example, `struct BadKey<'a>(&'a Cell<usize>)` will be seen as immutable
    /// and cause a false negative if its implementation of `Hash`/`Ord` accesses the `Cell`.
    ///
    /// This lint does check a few cases for indirection.  Firstly, using some standard library
    /// types (`Option`, `Result`, `Box`, `Rc`, `Arc`, `Vec`, `VecDeque`, `BTreeMap` and
    /// `BTreeSet`) directly as keys (e.g. in `HashMap<Box<Cell<usize>>, ()>`) **will** trigger the
    /// lint, because the impls of `Hash`/`Ord` for these types directly call `Hash`/`Ord` on their
    /// contained type.
    ///
    /// Secondly, the implementations of `Hash` and `Ord` for raw pointers (`*const T` or `*mut T`)
    /// apply only to the **address** of the contained value.  Therefore, interior mutability
    /// behind raw pointers (e.g. in `HashSet<*mut Cell<usize>>`) can't impact the value of `Hash`
    /// or `Ord`, and therefore will not trigger this link.  For more info, see issue
    /// [#6745](https://github.com/rust-lang/rust-clippy/issues/6745).
    ///
    /// ### Example
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
    #[clippy::version = "1.42.0"]
    pub MUTABLE_KEY_TYPE,
    suspicious,
    "Check for mutable `Map`/`Set` key type"
}

declare_lint_pass!(MutableKeyType => [ MUTABLE_KEY_TYPE ]);

impl<'tcx> LateLintPass<'tcx> for MutableKeyType {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        if let hir::ItemKind::Fn(ref sig, ..) = item.kind {
            check_sig(cx, item.hir_id(), sig.decl);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'tcx>) {
        if let hir::ImplItemKind::Fn(ref sig, ..) = item.kind {
            if trait_ref_of_method(cx, item.hir_id()).is_none() {
                check_sig(cx, item.hir_id(), sig.decl);
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Fn(ref sig, ..) = item.kind {
            check_sig(cx, item.hir_id(), sig.decl);
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
    for (hir_ty, ty) in iter::zip(decl.inputs, fn_sig.inputs().skip_binder()) {
        check_ty(cx, hir_ty.span, ty);
    }
    check_ty(cx, decl.output.span(), cx.tcx.erase_late_bound_regions(fn_sig.output()));
}

// We want to lint 1. sets or maps with 2. not immutable key types and 3. no unerased
// generics (because the compiler cannot ensure immutability for unknown types).
fn check_ty<'tcx>(cx: &LateContext<'tcx>, span: Span, ty: Ty<'tcx>) {
    let ty = ty.peel_refs();
    if let Adt(def, substs) = ty.kind() {
        let is_keyed_type = [sym::HashMap, sym::BTreeMap, sym::HashSet, sym::BTreeSet]
            .iter()
            .any(|diag_item| cx.tcx.is_diagnostic_item(*diag_item, def.did));
        if is_keyed_type && is_interior_mutable_type(cx, substs.type_at(0), span) {
            span_lint(cx, MUTABLE_KEY_TYPE, span, "mutable key type");
        }
    }
}

/// Determines if a type contains interior mutability which would affect its implementation of
/// [`Hash`] or [`Ord`].
fn is_interior_mutable_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, span: Span) -> bool {
    match *ty.kind() {
        Ref(_, inner_ty, mutbl) => mutbl == hir::Mutability::Mut || is_interior_mutable_type(cx, inner_ty, span),
        Slice(inner_ty) => is_interior_mutable_type(cx, inner_ty, span),
        Array(inner_ty, size) => {
            size.try_eval_usize(cx.tcx, cx.param_env).map_or(true, |u| u != 0)
                && is_interior_mutable_type(cx, inner_ty, span)
        },
        Tuple(..) => ty.tuple_fields().any(|ty| is_interior_mutable_type(cx, ty, span)),
        Adt(def, substs) => {
            // Special case for collections in `std` who's impl of `Hash` or `Ord` delegates to
            // that of their type parameters.  Note: we don't include `HashSet` and `HashMap`
            // because they have no impl for `Hash` or `Ord`.
            let is_std_collection = [
                sym::Option,
                sym::Result,
                sym::LinkedList,
                sym::Vec,
                sym::VecDeque,
                sym::BTreeMap,
                sym::BTreeSet,
                sym::Rc,
                sym::Arc,
            ]
            .iter()
            .any(|diag_item| cx.tcx.is_diagnostic_item(*diag_item, def.did));
            let is_box = Some(def.did) == cx.tcx.lang_items().owned_box();
            if is_std_collection || is_box {
                // The type is mutable if any of its type parameters are
                substs.types().any(|ty| is_interior_mutable_type(cx, ty, span))
            } else {
                !ty.has_escaping_bound_vars()
                    && cx.tcx.layout_of(cx.param_env.and(ty)).is_ok()
                    && !ty.is_freeze(cx.tcx.at(span), cx.param_env)
            }
        },
        _ => false,
    }
}
