//! Check the bodies of `const`s, `static`s and `const fn`s for illegal operations.
//!
//! This module will eventually replace the parts of `qualify_consts.rs` that check whether a local
//! has interior mutability or needs to be dropped, as well as the visitor that emits errors when
//! it finds operations that are invalid in a certain context.

use rustc_errors::DiagCtxtHandle;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, find_attr};
use rustc_middle::ty::{self, PolyFnSig, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_span::Symbol;

pub use self::qualifs::Qualif;

pub mod check;
mod ops;
pub mod post_drop_elaboration;
pub mod qualifs;
mod resolver;

/// Information about the item currently being const-checked, as well as a reference to the global
/// context.
pub struct ConstCx<'mir, 'tcx> {
    pub body: &'mir mir::Body<'tcx>,
    pub tcx: TyCtxt<'tcx>,
    pub typing_env: ty::TypingEnv<'tcx>,
    pub const_kind: Option<hir::ConstContext>,
}

impl<'mir, 'tcx> ConstCx<'mir, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'mir mir::Body<'tcx>) -> Self {
        let typing_env = body.typing_env(tcx);
        let const_kind = tcx.hir_body_const_context(body.source.def_id().expect_local());
        ConstCx { body, tcx, typing_env, const_kind }
    }

    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'tcx> {
        self.tcx.dcx()
    }

    pub fn def_id(&self) -> LocalDefId {
        self.body.source.def_id().expect_local()
    }

    /// Returns the kind of const context this `Item` represents (`const`, `static`, etc.).
    ///
    /// Panics if this `Item` is not const.
    pub fn const_kind(&self) -> hir::ConstContext {
        self.const_kind.expect("`const_kind` must not be called on a non-const fn")
    }

    pub fn enforce_recursive_const_stability(&self) -> bool {
        // We can skip this if neither `staged_api` nor `-Zforce-unstable-if-unmarked` are enabled,
        // since in such crates `lookup_const_stability` will always be `None`.
        self.const_kind == Some(hir::ConstContext::ConstFn)
            && (self.tcx.features().staged_api()
                || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked)
            && is_fn_or_trait_safe_to_expose_on_stable(self.tcx, self.def_id().to_def_id())
    }

    fn is_async(&self) -> bool {
        self.tcx.asyncness(self.def_id()).is_async()
    }

    pub fn fn_sig(&self) -> PolyFnSig<'tcx> {
        let did = self.def_id().to_def_id();
        if self.tcx.is_closure_like(did) {
            let ty = self.tcx.type_of(did).instantiate_identity();
            let ty::Closure(_, args) = ty.kind() else { bug!("type_of closure not ty::Closure") };
            args.as_closure().sig()
        } else {
            self.tcx.fn_sig(did).instantiate_identity()
        }
    }
}

pub fn rustc_allow_const_fn_unstable(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    feature_gate: Symbol,
) -> bool {
    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id));

    find_attr!(attrs, AttributeKind::AllowConstFnUnstable(syms, _) if syms.contains(&feature_gate))
}

/// Returns `true` if the given `def_id` (trait or function) is "safe to expose on stable".
///
/// This is relevant within a `staged_api` crate. Unlike with normal features, the use of unstable
/// const features *recursively* taints the functions that use them. This is to avoid accidentally
/// exposing e.g. the implementation of an unstable const intrinsic on stable. So we partition the
/// world into two functions: those that are safe to expose on stable (and hence may not use
/// unstable features, not even recursively), and those that are not.
pub fn is_fn_or_trait_safe_to_expose_on_stable(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // A default body in a `const trait` is const-stable when the trait is const-stable.
    if tcx.is_const_default_method(def_id) {
        return is_fn_or_trait_safe_to_expose_on_stable(tcx, tcx.parent(def_id));
    }

    match tcx.lookup_const_stability(def_id) {
        None => {
            // In a `staged_api` crate, we do enforce recursive const stability for all unmarked
            // functions, so we can trust local functions. But in another crate we don't know which
            // rules were applied, so we can't trust that.
            def_id.is_local() && tcx.features().staged_api()
        }
        Some(stab) => {
            // We consider things safe-to-expose if they are stable or if they are marked as
            // `const_stable_indirect`.
            stab.is_const_stable() || stab.const_stable_indirect
        }
    }
}
