use super::{Lift, Placeholder, TyCtxt};
use crate::ty;
use rustc_data_structures::intern::Interned;
use rustc_errors::ErrorGuaranteed;
use rustc_macros::HashStable;
use std::{fmt, marker::PhantomData};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Effect<'tcx>(pub Interned<'tcx, EffectData<'tcx>>);

impl<'tcx> std::ops::Deref for Effect<'tcx> {
    type Target = EffectData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}

impl<'tcx> fmt::Debug for Effect<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.0.fmt(f)
    }
}

impl<'tcx> Effect<'tcx> {
    pub fn is_e_infer(self) -> bool {
        matches!(self.val, ty::EffectValue::Infer(_))
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable, Debug, TyEncodable, TyDecodable)]
pub struct EffectData<'tcx> {
    pub val: EffectValue<'tcx>,
    pub kind: EffectKind,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable, Debug, TyEncodable, TyDecodable)]
#[derive(Copy)]
pub enum EffectValue<'tcx> {
    Rigid {
        on: bool,
    },
    /// Effect is forwarded from parent generics
    Param {
        index: u32,
    },
    /// Only used during type inference to decide on one of `On`/`Off`/`Param`
    Infer(InferEffect<'tcx>),

    /// Bound effect variable, used only when preparing a trait query.
    Bound(ty::DebruijnIndex, ty::BoundVar),

    /// A placeholder effect - universally quantified higher-ranked effect.
    Placeholder(PlaceholderEffect),

    /// An error occurred.
    Err(ErrorGuaranteed),
}

impl<'a, 'tcx> Lift<'tcx> for EffectValue<'a> {
    type Lifted = EffectValue<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<EffectValue<'tcx>> {
        Some(match self {
            Self::Rigid { on } => EffectValue::Rigid { on },
            Self::Param { index } => EffectValue::Param { index },
            Self::Infer(infer) => EffectValue::Infer(infer.lift_to_tcx(tcx)?),
            Self::Bound(di, bv) => EffectValue::Bound(di, bv),
            Self::Placeholder(ph) => EffectValue::Placeholder(ph),
            Self::Err(e) => EffectValue::Err(e),
        })
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable, Debug, TyEncodable, TyDecodable)]
#[derive(Copy, TypeFoldable, TypeVisitable)]
pub enum EffectKind {
    /// The opposite of `const`. This effect enables access to `static` variables, the file system,
    /// threads, networking, ...
    Host,
}

impl<'tcx> From<InferEffect<'tcx>> for EffectValue<'tcx> {
    fn from(infer: InferEffect<'tcx>) -> Self {
        Self::Infer(infer)
    }
}

impl<'tcx> From<EffectVid<'tcx>> for EffectValue<'tcx> {
    fn from(infer: EffectVid<'tcx>) -> Self {
        InferEffect::Var(infer).into()
    }
}

/// An inference variable for an effect, for use in effect generics.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Hash, Lift)]
#[derive(HashStable)]
pub enum InferEffect<'tcx> {
    /// Infer the value of the const.
    Var(EffectVid<'tcx>),
    /// A fresh effect variable. See `infer::freshen` for more details.
    Fresh(u32),
}

/// An **effect** **v**ariable **ID**.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub struct EffectVid<'tcx> {
    pub index: u32,
    pub phantom: PhantomData<&'tcx ()>,
}

impl<'a, 'tcx> Lift<'tcx> for EffectVid<'a> {
    type Lifted = EffectVid<'tcx>;
    fn lift_to_tcx(self, _tcx: TyCtxt<'tcx>) -> Option<EffectVid<'tcx>> {
        Some(EffectVid { index: self.index, phantom: PhantomData })
    }
}

pub type PlaceholderEffect = Placeholder<ty::BoundVar>;
