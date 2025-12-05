//! Unification keyes for the infer context the next-trait-solver.

use std::cmp;
use std::marker::PhantomData;

use ena::unify::{NoError, UnifyKey, UnifyValue};
use rustc_type_ir::{ConstVid, RegionKind, RegionVid, UniverseIndex, inherent::IntoKind};

use crate::next_solver::{Const, Region};

#[derive(Clone, Debug)]
pub(crate) enum RegionVariableValue<'db> {
    Known { value: Region<'db> },
    Unknown { universe: UniverseIndex },
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

            (RegionVariableValue::Known { value }, RegionVariableValue::Unknown { universe })
            | (RegionVariableValue::Unknown { universe }, RegionVariableValue::Known { value }) => {
                let universe_of_value = match (*value).kind() {
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

                if universe.can_name(universe_of_value) {
                    Ok(RegionVariableValue::Known { value: *value })
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
                Ok(RegionVariableValue::Unknown { universe: (*a).min(*b) })
            }
        }
    }
}

// Generic consts.

#[derive(Copy, Clone, Debug)]
pub struct ConstVariableOrigin {}

#[derive(Clone, Debug)]
pub(crate) enum ConstVariableValue<'db> {
    Known { value: Const<'db> },
    Unknown { origin: ConstVariableOrigin, universe: UniverseIndex },
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
    fn order_roots(a: Self, _: &Self::Value, b: Self, _: &Self::Value) -> Option<(Self, Self)> {
        if a.vid.as_u32() < b.vid.as_u32() { Some((a, b)) } else { Some((b, a)) }
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
            (ConstVariableValue::Known { .. }, ConstVariableValue::Unknown { .. }) => {
                Ok(value1.clone())
            }
            (ConstVariableValue::Unknown { .. }, ConstVariableValue::Known { .. }) => {
                Ok(value2.clone())
            }

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
                let universe = cmp::min(*universe1, *universe2);
                Ok(ConstVariableValue::Unknown { origin: *origin, universe })
            }
        }
    }
}
