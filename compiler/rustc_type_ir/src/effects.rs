use crate::lang_items::TraitSolverLangItem::{EffectsMaybe, EffectsRuntime, EffectsNoRuntime};
use crate::Interner;
use crate::inherent::{AdtDef, IntoKind, Ty};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    Maybe,
    Runtime,
    NoRuntime,
}

impl EffectKind {
    pub fn try_from_def_id<I: Interner>(tcx: I, def_id: I::DefId) -> Option<EffectKind> {
        if tcx.is_lang_item(def_id, EffectsMaybe) {
            Some(EffectKind::Maybe)
        } else if tcx.is_lang_item(def_id, EffectsRuntime) {
            Some(EffectKind::Runtime)
        } else if tcx.is_lang_item(def_id, EffectsNoRuntime) {
            Some(EffectKind::NoRuntime)
        } else {
            None
        }
    }

    pub fn to_def_id<I: Interner>(self, tcx: I) -> I::DefId {
        let lang_item = match self {
            EffectKind::Maybe => EffectsMaybe,
            EffectKind::NoRuntime => EffectsNoRuntime,
            EffectKind::Runtime => EffectsRuntime,
        };

        tcx.require_lang_item(lang_item)
    }

    pub fn try_from_ty<I: Interner>(tcx: I, ty: I::Ty) -> Option<EffectKind> {
        if let crate::Adt(def, _) = ty.kind() {
            Self::try_from_def_id(tcx, def.def_id())
        } else {
            None
        }
    }

    pub fn to_ty<I: Interner>(self, tcx: I) -> I::Ty {
        I::Ty::new_adt(tcx, tcx.adt_def(self.to_def_id(tcx)), Default::default())
    }

    pub fn min(a: Self, b: Self) -> Option<Self> {
        use EffectKind::*;
        match (a, b) {
            (Maybe, x) | (x, Maybe) => Some(x),
            (Runtime, Runtime) => Some(Runtime),
            (NoRuntime, NoRuntime) => Some(NoRuntime),
            (Runtime, NoRuntime) | (NoRuntime, Runtime) => None,
        }
    }
}