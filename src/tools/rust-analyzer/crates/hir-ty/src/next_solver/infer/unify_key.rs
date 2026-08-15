//! Unification keyes for the infer context the next-trait-solver.

use std::cmp;
use std::marker::PhantomData;

use ena::unify::{NoError, UnifyKey, UnifyValue};
use rustc_type_ir::{ConstVid, RegionKind, RegionVid, UniverseIndex, inherent::IntoKind};

use crate::{
    Span,
    next_solver::{Const, Region},
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum RegionVariableValue<'db> {
    Known { value: Region<'db>, span: Option<Span> },
    Unknown { universe: UniverseIndex, span: Span },
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub(crate) struct RegionVidKey<'db> {
    pub(crate) vid: RegionVid,
    pub(crate) phantom: PhantomData<RegionVariableValue<'db>>,
}

impl<'db> From<RegionVid> for RegionVidKey<'db> {
    fn from(vid: RegionVid) -> Self {
        RegionVidKey { vid, phantom: PhantomData }
    }
}

impl<'db> UnifyKey for RegionVidKey<'db> {
    type Value = RegionVariableValue<'db>;
    #[inline]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        RegionVidKey::from(RegionVid::from_u32(i))
    }
    fn tag() -> &'static str {
        "RegionVidKey"
    }
}

pub(crate) struct RegionUnificationError;
impl<'db> UnifyValue for RegionVariableValue<'db> {
    type Error = RegionUnificationError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        match (value1, value2) {
            (RegionVariableValue::Known { .. }, RegionVariableValue::Known { .. }) => {
                Err(RegionUnificationError)
            }

            (
                &RegionVariableValue::Known { value, span: span_known },
                &RegionVariableValue::Unknown { universe, span: span_unknown },
            )
            | (
                &RegionVariableValue::Unknown { universe, span: span_unknown },
                &RegionVariableValue::Known { value, span: span_known },
            ) => {
                let universe_of_value = match value.kind() {
                    RegionKind::ReStatic
                    | RegionKind::ReErased
                    | RegionKind::ReLateParam(..)
                    | RegionKind::ReEarlyParam(..)
                    | RegionKind::ReError(_) => UniverseIndex::ROOT,
                    RegionKind::RePlaceholder(placeholder) => placeholder.universe,
                    RegionKind::ReVar(..) | RegionKind::ReBound(..) => {
                        panic!("not a universal region")
                    }
                };

                let span = match span_known {
                    Some(span_known) => Span::pick_best(span_known, span_unknown),
                    None => span_unknown,
                };
                if universe.can_name(universe_of_value) {
                    Ok(RegionVariableValue::Known { value, span: Some(span) })
                } else {
                    Err(RegionUnificationError)
                }
            }

            (
                &RegionVariableValue::Unknown { universe: a, span: span1 },
                &RegionVariableValue::Unknown { universe: b, span: span2 },
            ) => {
                // If we unify two unconstrained regions then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                let span = Span::pick_best(span1, span2);
                Ok(RegionVariableValue::Unknown { universe: a.min(b), span })
            }
        }
    }
}

// Generic consts.

#[derive(Clone, Copy, Debug)]
pub(crate) enum ConstVariableValue<'db> {
    Known { value: Const<'db> },
    Unknown { span: Span, universe: UniverseIndex },
}

impl<'db> ConstVariableValue<'db> {
    /// If this value is known, returns the const it is known to be.
    /// Otherwise, `None`.
    pub(crate) fn known(&self) -> Option<Const<'db>> {
        match self {
            ConstVariableValue::Unknown { .. } => None,
            ConstVariableValue::Known { value } => Some(*value),
        }
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub(crate) struct ConstVidKey<'db> {
    pub(crate) vid: ConstVid,
    pub(crate) phantom: PhantomData<Const<'db>>,
}

impl<'db> From<ConstVid> for ConstVidKey<'db> {
    fn from(vid: ConstVid) -> Self {
        ConstVidKey { vid, phantom: PhantomData }
    }
}

impl<'db> UnifyKey for ConstVidKey<'db> {
    type Value = ConstVariableValue<'db>;
    #[inline]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        ConstVidKey::from(ConstVid::from_u32(i))
    }
    fn tag() -> &'static str {
        "ConstVidKey"
    }
}

impl<'db> UnifyValue for ConstVariableValue<'db> {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        match (value1, value2) {
            (ConstVariableValue::Known { .. }, ConstVariableValue::Known { .. }) => {
                panic!("equating two const variables, both of which have known values")
            }

            // If one side is known, prefer that one.
            (ConstVariableValue::Known { .. }, ConstVariableValue::Unknown { .. }) => Ok(*value1),
            (ConstVariableValue::Unknown { .. }, ConstVariableValue::Known { .. }) => Ok(*value2),

            // If both sides are *unknown*, it hardly matters, does it?
            (
                &ConstVariableValue::Unknown { span: span1, universe: universe1 },
                &ConstVariableValue::Unknown { span: span2, universe: universe2 },
            ) => {
                // If we unify two unbound variables, ?T and ?U, then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                let universe = cmp::min(universe1, universe2);
                let span = Span::pick_best(span1, span2);
                Ok(ConstVariableValue::Unknown { span, universe })
            }
        }
    }
}
