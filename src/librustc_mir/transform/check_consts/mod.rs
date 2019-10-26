//! Check the bodies of `const`s, `static`s and `const fn`s for illegal operations.
//!
//! This module will eventually replace the parts of `qualify_consts.rs` that check whether a local
//! has interior mutability or needs to be dropped, as well as the visitor that emits errors when
//! it finds operations that are invalid in a certain context.

use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::ty::{self, TyCtxt};

pub use self::qualifs::Qualif;

pub mod ops;
pub mod qualifs;
mod resolver;
pub mod validation;

/// Information about the item currently being validated, as well as a reference to the global
/// context.
pub struct Item<'mir, 'tcx> {
    body: &'mir mir::Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    param_env: ty::ParamEnv<'tcx>,
    mode: validation::Mode,
    for_promotion: bool,
}

impl Item<'mir, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        body: &'mir mir::Body<'tcx>,
    ) -> Self {
        let param_env = tcx.param_env(def_id);
        let mode = validation::Mode::for_item(tcx, def_id)
            .expect("const validation must only be run inside a const context");

        Item {
            body,
            tcx,
            def_id,
            param_env,
            mode,
            for_promotion: false,
        }
    }

    // HACK(eddyb) this is to get around the panic for a runtime fn from `Item::new`.
    // Also, it allows promoting `&mut []`.
    pub fn for_promotion(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        body: &'mir mir::Body<'tcx>,
    ) -> Self {
        let param_env = tcx.param_env(def_id);
        let mode = validation::Mode::for_item(tcx, def_id)
            .unwrap_or(validation::Mode::ConstFn);

        Item {
            body,
            tcx,
            def_id,
            param_env,
            mode,
            for_promotion: true,
        }
    }
}


fn is_lang_panic_fn(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    Some(def_id) == tcx.lang_items().panic_fn() ||
    Some(def_id) == tcx.lang_items().begin_panic_fn()
}
