use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::trait_ref_of_method;
use clippy_utils::ty::InteriorMut;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
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
    /// It's correct to use a struct that contains interior mutability as a key when its
    /// implementation of `Hash` or `Ord` doesn't access any of the interior mutable types.
    /// However, this lint is unable to recognize this, so it will often cause false positives in
    /// these cases.
    ///
    /// #### False Negatives
    /// This lint does not follow raw pointers (`*const T` or `*mut T`) as `Hash` and `Ord`
    /// apply only to the **address** of the contained value. This can cause false negatives for
    /// custom collections that use raw pointers internally.
    ///
    /// ### Example
    /// ```no_run
    /// use std::cmp::{PartialEq, Eq};
    /// use std::collections::HashSet;
    /// use std::hash::{Hash, Hasher};
    /// use std::sync::atomic::AtomicUsize;
    ///
    /// struct Bad(AtomicUsize);
    /// impl PartialEq for Bad {
    ///     fn eq(&self, rhs: &Self) -> bool {
    ///          ..
    /// # ; true
    ///     }
    /// }
    ///
    /// impl Eq for Bad {}
    ///
    /// impl Hash for Bad {
    ///     fn hash<H: Hasher>(&self, h: &mut H) {
    ///         ..
    /// # ;
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

pub struct MutableKeyType<'tcx> {
    interior_mut: InteriorMut<'tcx>,
}

impl_lint_pass!(MutableKeyType<'_> => [ MUTABLE_KEY_TYPE ]);

impl<'tcx> LateLintPass<'tcx> for MutableKeyType<'tcx> {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        if let hir::ItemKind::Fn { ref sig, .. } = item.kind {
            self.check_sig(cx, item.owner_id.def_id, sig.decl);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'tcx>) {
        if let hir::ImplItemKind::Fn(ref sig, ..) = item.kind
            && trait_ref_of_method(cx, item.owner_id.def_id).is_none()
        {
            self.check_sig(cx, item.owner_id.def_id, sig.decl);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Fn(ref sig, ..) = item.kind {
            self.check_sig(cx, item.owner_id.def_id, sig.decl);
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &hir::LetStmt<'tcx>) {
        if let hir::PatKind::Wild = local.pat.kind {
            return;
        }
        self.check_ty_(cx, local.span, cx.typeck_results().pat_ty(local.pat));
    }
}

impl<'tcx> MutableKeyType<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, conf: &'static Conf) -> Self {
        Self {
            interior_mut: InteriorMut::without_pointers(tcx, &conf.ignore_interior_mutability),
        }
    }

    fn check_sig(&mut self, cx: &LateContext<'tcx>, fn_def_id: LocalDefId, decl: &hir::FnDecl<'tcx>) {
        let fn_sig = cx.tcx.fn_sig(fn_def_id).instantiate_identity();
        for (hir_ty, ty) in iter::zip(decl.inputs, fn_sig.inputs().skip_binder()) {
            self.check_ty_(cx, hir_ty.span, *ty);
        }
        self.check_ty_(
            cx,
            decl.output.span(),
            cx.tcx.instantiate_bound_regions_with_erased(fn_sig.output()),
        );
    }

    // We want to lint 1. sets or maps with 2. not immutable key types and 3. no unerased
    // generics (because the compiler cannot ensure immutability for unknown types).
    fn check_ty_(&mut self, cx: &LateContext<'tcx>, span: Span, ty: Ty<'tcx>) {
        let ty = ty.peel_refs();
        if let ty::Adt(def, args) = ty.kind()
            && matches!(
                cx.tcx.get_diagnostic_name(def.did()),
                Some(sym::HashMap | sym::BTreeMap | sym::HashSet | sym::BTreeSet)
            )
        {
            let subst_ty = args.type_at(0);
            if let Some(chain) = self.interior_mut.interior_mut_ty_chain(cx, subst_ty) {
                span_lint_and_then(cx, MUTABLE_KEY_TYPE, span, "mutable key type", |diag| {
                    for ty in chain.iter().rev() {
                        diag.note(with_forced_trimmed_paths!(format!(
                            "... because it contains `{ty}`, which has interior mutability"
                        )));
                    }
                });
            }
        }
    }
}
