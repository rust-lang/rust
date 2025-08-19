use std::cmp;
use std::marker::PhantomData;

use rustc_data_structures::unify::{NoError, UnifyKey, UnifyValue};
use rustc_middle::{bug, ty};
use rustc_span::Span;
use rustc_span::def_id::DefId;

#[derive(Copy, Clone, Debug)]
pub(crate) enum RegionVariableValue<'tcx> {
    Known { value: ty::Region<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub(crate) struct RegionVidKey<'tcx> {
    pub vid: ty::RegionVid,
    pub phantom: PhantomData<RegionVariableValue<'tcx>>,
}

impl<'tcx> From<ty::RegionVid> for RegionVidKey<'tcx> {
    fn from(vid: ty::RegionVid) -> Self {
        RegionVidKey { vid, phantom: PhantomData }
    }
}

impl<'tcx> UnifyKey for RegionVidKey<'tcx> {
    type Value = RegionVariableValue<'tcx>;
    #[inline]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        RegionVidKey::from(ty::RegionVid::from_u32(i))
    }
    fn tag() -> &'static str {
        "RegionVidKey"
    }
}

pub(crate) struct RegionUnificationError;

impl<'tcx> UnifyValue for RegionVariableValue<'tcx> {
    type Error = RegionUnificationError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        match (*value1, *value2) {
            (RegionVariableValue::Known { .. }, RegionVariableValue::Known { .. }) => {
                Err(RegionUnificationError)
            }

            (RegionVariableValue::Known { value }, RegionVariableValue::Unknown { universe })
            | (RegionVariableValue::Unknown { universe }, RegionVariableValue::Known { value }) => {
                let universe_of_value = match value.kind() {
                    ty::ReStatic
                    | ty::ReErased
                    | ty::ReLateParam(..)
                    | ty::ReEarlyParam(..)
                    | ty::ReError(_) => ty::UniverseIndex::ROOT,
                    ty::RePlaceholder(placeholder) => placeholder.universe,
                    ty::ReVar(..) | ty::ReBound(..) => bug!("not a universal region"),
                };

                if universe.can_name(universe_of_value) {
                    Ok(RegionVariableValue::Known { value })
                } else {
                    Err(RegionUnificationError)
                }
            }

            (
                RegionVariableValue::Unknown { universe: a },
                RegionVariableValue::Unknown { universe: b },
            ) => {
                // If we unify two unconstrained regions then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                Ok(RegionVariableValue::Unknown { universe: a.min(b) })
            }
        }
    }
}

// Generic consts.

#[derive(Copy, Clone, Debug)]
pub struct ConstVariableOrigin {
    pub span: Span,
    /// `DefId` of the const parameter this was instantiated for, if any.
    ///
    /// This should only be used for diagnostics.
    pub param_def_id: Option<DefId>,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum ConstVariableValue<'tcx> {
    Known { value: ty::Const<'tcx> },
    Unknown { origin: ConstVariableOrigin, universe: ty::UniverseIndex },
}

impl<'tcx> ConstVariableValue<'tcx> {
    /// If this value is known, returns the const it is known to be.
    /// Otherwise, `None`.
    pub(crate) fn known(&self) -> Option<ty::Const<'tcx>> {
        match *self {
            ConstVariableValue::Unknown { .. } => None,
            ConstVariableValue::Known { value } => Some(value),
        }
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub(crate) struct ConstVidKey<'tcx> {
    pub vid: ty::ConstVid,
    pub phantom: PhantomData<ty::Const<'tcx>>,
}

impl<'tcx> From<ty::ConstVid> for ConstVidKey<'tcx> {
    fn from(vid: ty::ConstVid) -> Self {
        ConstVidKey { vid, phantom: PhantomData }
    }
}

impl<'tcx> UnifyKey for ConstVidKey<'tcx> {
    type Value = ConstVariableValue<'tcx>;
    #[inline]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        ConstVidKey::from(ty::ConstVid::from_u32(i))
    }
    fn tag() -> &'static str {
        "ConstVidKey"
    }
    fn order_roots(a: Self, _: &Self::Value, b: Self, _: &Self::Value) -> Option<(Self, Self)> {
        if a.vid.as_u32() < b.vid.as_u32() { Some((a, b)) } else { Some((b, a)) }
    }
}

impl<'tcx> UnifyValue for ConstVariableValue<'tcx> {
    type Error = NoError;

    fn unify_values(&value1: &Self, &value2: &Self) -> Result<Self, Self::Error> {
        match (value1, value2) {
            (ConstVariableValue::Known { .. }, ConstVariableValue::Known { .. }) => {
                bug!("equating two const variables, both of which have known values")
            }

            // If one side is known, prefer that one.
            (ConstVariableValue::Known { .. }, ConstVariableValue::Unknown { .. }) => Ok(value1),
            (ConstVariableValue::Unknown { .. }, ConstVariableValue::Known { .. }) => Ok(value2),

            // If both sides are *unknown*, it hardly matters, does it?
            (
                ConstVariableValue::Unknown { origin, universe: universe1 },
                ConstVariableValue::Unknown { origin: _, universe: universe2 },
            ) => {
                // If we unify two unbound variables, ?T and ?U, then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                let universe = cmp::min(universe1, universe2);
                Ok(ConstVariableValue::Unknown { origin, universe })
            }
        }
    }
}
