//! Check the bodies of `const`s, `static`s and `const fn`s for illegal operations.
//!
//! This module will eventually replace the parts of `qualify_consts.rs` that check whether a local
//! has interior mutability or needs to be dropped, as well as the visitor that emits errors when
//! it finds operations that are invalid in a certain context.

use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir;
use rustc_middle::ty::{self, TyCtxt};

pub use self::qualifs::Qualif;

mod ops;
pub mod qualifs;
mod resolver;
pub mod validation;

/// Information about the item currently being const-checked, as well as a reference to the global
/// context.
pub struct ConstCx<'mir, 'tcx> {
    pub body: &'mir mir::Body<'tcx>,
    pub tcx: TyCtxt<'tcx>,
    pub def_id: DefId,
    pub param_env: ty::ParamEnv<'tcx>,
    pub const_kind: Option<hir::ConstContext>,
}

impl ConstCx<'mir, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &'mir mir::Body<'tcx>) -> Self {
        let param_env = tcx.param_env(def_id);
        Self::new_with_param_env(tcx, def_id, body, param_env)
    }

    pub fn new_with_param_env(
        tcx: TyCtxt<'tcx>,
        def_id: LocalDefId,
        body: &'mir mir::Body<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let const_kind = tcx.hir().body_const_context(def_id);
        ConstCx { body, tcx, def_id: def_id.to_def_id(), param_env, const_kind }
    }

    /// Returns the kind of const context this `Item` represents (`const`, `static`, etc.).
    ///
    /// Panics if this `Item` is not const.
    pub fn const_kind(&self) -> hir::ConstContext {
        self.const_kind.expect("`const_kind` must not be called on a non-const fn")
    }
}

/// Returns `true` if this `DefId` points to one of the official `panic` lang items.
pub fn is_lang_panic_fn(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    Some(def_id) == tcx.lang_items().panic_fn() || Some(def_id) == tcx.lang_items().begin_panic_fn()
}
