//! Check the bodies of `const`s, `static`s and `const fn`s for illegal operations.
//!
//! This module will eventually replace the parts of `qualify_consts.rs` that check whether a local
//! has interior mutability or needs to be dropped, as well as the visitor that emits errors when
//! it finds operations that are invalid in a certain context.

use rustc::hir::{self, def_id::DefId};
use rustc::mir;
use rustc::ty::{self, TyCtxt};

use std::fmt;

pub use self::qualifs::Qualif;

pub mod ops;
pub mod qualifs;
mod resolver;
pub mod validation;

/// Information about the item currently being const-checked, as well as a reference to the global
/// context.
pub struct Item<'mir, 'tcx> {
    pub body: mir::ReadOnlyBodyAndCache<'mir, 'tcx>,
    pub tcx: TyCtxt<'tcx>,
    pub def_id: DefId,
    pub param_env: ty::ParamEnv<'tcx>,
    pub const_kind: Option<ConstKind>,
}

impl Item<'mir, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        body: mir::ReadOnlyBodyAndCache<'mir, 'tcx>,
    ) -> Self {
        let param_env = tcx.param_env(def_id);
        let const_kind = ConstKind::for_item(tcx, def_id);

        Item {
            body,
            tcx,
            def_id,
            param_env,
            const_kind,
        }
    }

    /// Returns the kind of const context this `Item` represents (`const`, `static`, etc.).
    ///
    /// Panics if this `Item` is not const.
    pub fn const_kind(&self) -> ConstKind {
        self.const_kind.expect("`const_kind` must not be called on a non-const fn")
    }
}

/// The kinds of items which require compile-time evaluation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConstKind {
    /// A `static` item.
    Static,
    /// A `static mut` item.
    StaticMut,
    /// A `const fn` item.
    ConstFn,
    /// A `const` item or an anonymous constant (e.g. in array lengths).
    Const,
}

impl ConstKind {
    /// Returns the validation mode for the item with the given `DefId`, or `None` if this item
    /// does not require validation (e.g. a non-const `fn`).
    pub fn for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<Self> {
        use hir::BodyOwnerKind as HirKind;

        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

        let mode = match tcx.hir().body_owner_kind(hir_id) {
            HirKind::Closure => return None,

            // Note: this is deliberately checking for `is_const_fn_raw`, as the `is_const_fn`
            // checks take into account the `rustc_const_unstable` attribute combined with enabled
            // feature gates. Otherwise, const qualification would _not check_ whether this
            // function body follows the `const fn` rules, as an unstable `const fn` would
            // be considered "not const". More details are available in issue #67053.
            HirKind::Fn if tcx.is_const_fn_raw(def_id) => ConstKind::ConstFn,
            HirKind::Fn => return None,

            HirKind::Const => ConstKind::Const,

            HirKind::Static(hir::Mutability::Not) => ConstKind::Static,
            HirKind::Static(hir::Mutability::Mut) => ConstKind::StaticMut,
        };

        Some(mode)
    }

    pub fn is_static(self) -> bool {
        match self {
            ConstKind::Static | ConstKind::StaticMut => true,
            ConstKind::ConstFn | ConstKind::Const => false,
        }
    }
}

impl fmt::Display for ConstKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ConstKind::Const => write!(f, "constant"),
            ConstKind::Static | ConstKind::StaticMut => write!(f, "static"),
            ConstKind::ConstFn => write!(f, "constant function"),
        }
    }
}

/// Returns `true` if this `DefId` points to one of the official `panic` lang items.
pub fn is_lang_panic_fn(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    Some(def_id) == tcx.lang_items().panic_fn() ||
    Some(def_id) == tcx.lang_items().begin_panic_fn()
}
