//! Check the bodies of `const`s, `static`s and `const fn`s for illegal operations.
//!
//! This module will eventually replace the parts of `qualify_consts.rs` that check whether a local
//! has interior mutability or needs to be dropped, as well as the visitor that emits errors when
//! it finds operations that are invalid in a certain context.

use rustc_attr as attr;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir;
use rustc_middle::ty::{self, PolyFnSig, TyCtxt};
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
    pub param_env: ty::ParamEnv<'tcx>,
    pub const_kind: Option<hir::ConstContext>,
}

impl<'mir, 'tcx> ConstCx<'mir, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'mir mir::Body<'tcx>) -> Self {
        let def_id = body.source.def_id().expect_local();
        let param_env = tcx.param_env(def_id);
        Self::new_with_param_env(tcx, body, param_env)
    }

    pub fn new_with_param_env(
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let const_kind = tcx.hir().body_const_context(body.source.def_id().expect_local());
        ConstCx { body, tcx, param_env, const_kind }
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

    pub fn is_const_stable_const_fn(&self) -> bool {
        self.const_kind == Some(hir::ConstContext::ConstFn)
            && self.tcx.features().staged_api
            && is_const_stable_const_fn(self.tcx, self.def_id().to_def_id())
    }

    fn is_async(&self) -> bool {
        self.tcx.asyncness(self.def_id()).is_async()
    }

    pub fn fn_sig(&self) -> PolyFnSig<'tcx> {
        let did = self.def_id().to_def_id();
        if self.tcx.is_closure(did) {
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
    let attrs = tcx.hir().attrs(tcx.hir().local_def_id_to_hir_id(def_id));
    attr::rustc_allow_const_fn_unstable(&tcx.sess, attrs).any(|name| name == feature_gate)
}

/// Returns `true` if the given `const fn` is "const-stable".
///
/// Panics if the given `DefId` does not refer to a `const fn`.
///
/// Const-stability is only relevant for `const fn` within a `staged_api` crate. Only "const-stable"
/// functions can be called in a const-context by users of the stable compiler. "const-stable"
/// functions are subject to more stringent restrictions than "const-unstable" functions: They
/// cannot use unstable features and can only call other "const-stable" functions.
pub fn is_const_stable_const_fn(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // A default body in a `#[const_trait]` is not const-stable because const
    // trait fns currently cannot be const-stable. We shouldn't
    // restrict default bodies to only call const-stable functions.
    if tcx.is_const_default_method(def_id) {
        return false;
    }

    // Const-stability is only relevant for `const fn`.
    assert!(tcx.is_const_fn_raw(def_id));

    // A function is only const-stable if it has `#[rustc_const_stable]` or it the trait it belongs
    // to is const-stable.
    match tcx.lookup_const_stability(def_id) {
        Some(stab) => stab.is_const_stable(),
        None if is_parent_const_stable_trait(tcx, def_id) => {
            // Remove this when `#![feature(const_trait_impl)]` is stabilized,
            // returning `true` unconditionally.
            tcx.sess.delay_span_bug(
                tcx.def_span(def_id),
                "trait implementations cannot be const stable yet",
            );
            true
        }
        None => false, // By default, items are not const stable.
    }
}

fn is_parent_const_stable_trait(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let local_def_id = def_id.expect_local();
    let hir_id = tcx.local_def_id_to_hir_id(local_def_id);

    let Some(parent) = tcx.hir().opt_parent_id(hir_id) else { return false };
    let parent_def = tcx.hir().get(parent);

    if !matches!(
        parent_def,
        hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Impl(hir::Impl { constness: hir::Constness::Const, .. }),
            ..
        })
    ) {
        return false;
    }

    tcx.lookup_const_stability(parent.owner).is_some_and(|stab| stab.is_const_stable())
}
