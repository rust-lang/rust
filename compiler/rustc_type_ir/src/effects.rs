use crate::Interner;
use crate::inherent::*;
use crate::lang_items::TraitSolverLangItem::{EffectsMaybe, EffectsNoRuntime, EffectsRuntime};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    Maybe,
    Runtime,
    NoRuntime,
}

impl EffectKind {
    pub fn try_from_def_id<I: Interner>(cx: I, def_id: I::DefId) -> Option<EffectKind> {
        if cx.is_lang_item(def_id, EffectsMaybe) {
            Some(EffectKind::Maybe)
        } else if cx.is_lang_item(def_id, EffectsRuntime) {
            Some(EffectKind::Runtime)
        } else if cx.is_lang_item(def_id, EffectsNoRuntime) {
            Some(EffectKind::NoRuntime)
        } else {
            None
        }
    }

    pub fn to_def_id<I: Interner>(self, cx: I) -> I::DefId {
        let lang_item = match self {
            EffectKind::Maybe => EffectsMaybe,
            EffectKind::NoRuntime => EffectsNoRuntime,
            EffectKind::Runtime => EffectsRuntime,
        };

        cx.require_lang_item(lang_item)
    }

    pub fn try_from_ty<I: Interner>(cx: I, ty: I::Ty) -> Option<EffectKind> {
        if let crate::Adt(def, _) = ty.kind() {
            Self::try_from_def_id(cx, def.def_id())
        } else {
            None
        }
    }

    pub fn to_ty<I: Interner>(self, cx: I) -> I::Ty {
        I::Ty::new_adt(cx, cx.adt_def(self.to_def_id(cx)), Default::default())
    }

    /// Returns an intersection between two effect kinds. If one effect kind
    /// is more permissive than the other (e.g. `Maybe` vs `Runtime`), this
    /// returns the less permissive effect kind (`Runtime`).
    pub fn intersection(a: Self, b: Self) -> Option<Self> {
        use EffectKind::*;
        match (a, b) {
            (Maybe, x) | (x, Maybe) => Some(x),
            (Runtime, Runtime) => Some(Runtime),
            (NoRuntime, NoRuntime) => Some(NoRuntime),
            (Runtime, NoRuntime) | (NoRuntime, Runtime) => None,
        }
    }
}
