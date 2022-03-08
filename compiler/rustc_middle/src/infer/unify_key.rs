use crate::ty::{self, InferConst, Ty, TyCtxt};
use rustc_data_structures::snapshot_vec;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_data_structures::unify::{
    self, EqUnifyValue, InPlace, NoError, UnificationTable, UnifyKey, UnifyValue,
};
use rustc_span::def_id::DefId;
use rustc_span::symbol::Symbol;
use rustc_span::Span;

use std::cmp;
use std::marker::PhantomData;

pub trait ToType {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct UnifiedRegion<'tcx>(pub Option<ty::Region<'tcx>>);

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct RegionVidKey<'tcx> {
    pub vid: ty::RegionVid,
    pub phantom: PhantomData<UnifiedRegion<'tcx>>,
}

impl<'tcx> From<ty::RegionVid> for RegionVidKey<'tcx> {
    fn from(vid: ty::RegionVid) -> Self {
        RegionVidKey { vid, phantom: PhantomData }
    }
}

impl<'tcx> UnifyKey for RegionVidKey<'tcx> {
    type Value = UnifiedRegion<'tcx>;
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

impl<'tcx> UnifyValue for UnifiedRegion<'tcx> {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, NoError> {
        Ok(match (value1.0, value2.0) {
            // Here we can just pick one value, because the full constraints graph
            // will be handled later. Ideally, we might want a `MultipleValues`
            // variant or something. For now though, this is fine.
            (Some(_), Some(_)) => *value1,

            (Some(_), _) => *value1,
            (_, Some(_)) => *value2,

            (None, None) => *value1,
        })
    }
}

impl ToType for ty::IntVarValue {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => tcx.mk_mach_int(i),
            ty::UintType(i) => tcx.mk_mach_uint(i),
        }
    }
}

impl ToType for ty::FloatVarValue {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        tcx.mk_mach_float(self.0)
    }
}

// Generic consts.

#[derive(Copy, Clone, Debug)]
pub struct ConstVariableOrigin {
    pub kind: ConstVariableOriginKind,
    pub span: Span,
}

/// Reasons to create a const inference variable
#[derive(Copy, Clone, Debug)]
pub enum ConstVariableOriginKind {
    MiscVariable,
    ConstInference,
    ConstParameterDefinition(Symbol, DefId),
    SubstitutionPlaceholder,
}

#[derive(Copy, Clone, Debug)]
pub enum ConstVariableValue<'tcx> {
    Known { value: ty::Const<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

impl<'tcx> ConstVariableValue<'tcx> {
    /// If this value is known, returns the const it is known to be.
    /// Otherwise, `None`.
    pub fn known(&self) -> Option<ty::Const<'tcx>> {
        match *self {
            ConstVariableValue::Unknown { .. } => None,
            ConstVariableValue::Known { value } => Some(value),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ConstVarValue<'tcx> {
    pub origin: ConstVariableOrigin,
    pub val: ConstVariableValue<'tcx>,
}

impl<'tcx> UnifyKey for ty::ConstVid<'tcx> {
    type Value = ConstVarValue<'tcx>;
    #[inline]
    fn index(&self) -> u32 {
        self.index
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        ty::ConstVid { index: i, phantom: PhantomData }
    }
    fn tag() -> &'static str {
        "ConstVid"
    }
}

impl<'tcx> UnifyValue for ConstVarValue<'tcx> {
    type Error = (ty::Const<'tcx>, ty::Const<'tcx>);

    fn unify_values(&value1: &Self, &value2: &Self) -> Result<Self, Self::Error> {
        Ok(match (value1.val, value2.val) {
            (ConstVariableValue::Known { .. }, ConstVariableValue::Known { .. }) => {
                bug!("equating two const variables, both of which have known values")
            }

            // If one side is known, prefer that one.
            (ConstVariableValue::Known { .. }, ConstVariableValue::Unknown { .. }) => value1,
            (ConstVariableValue::Unknown { .. }, ConstVariableValue::Known { .. }) => value2,

            // If both sides are *unknown*, it hardly matters, does it?
            (
                ConstVariableValue::Unknown { universe: universe1 },
                ConstVariableValue::Unknown { universe: universe2 },
            ) => {
                // If we unify two unbound variables, ?T and ?U, then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                let universe = cmp::min(universe1, universe2);
                ConstVarValue {
                    val: ConstVariableValue::Unknown { universe },
                    origin: value1.origin,
                }
            }
        })
    }
}

impl<'tcx> EqUnifyValue for ty::Const<'tcx> {}

pub fn replace_if_possible<'tcx, V, L>(
    table: &mut UnificationTable<InPlace<ty::ConstVid<'tcx>, V, L>>,
    c: ty::Const<'tcx>,
) -> ty::Const<'tcx>
where
    V: snapshot_vec::VecLike<unify::Delegate<ty::ConstVid<'tcx>>>,
    L: UndoLogs<snapshot_vec::UndoLog<unify::Delegate<ty::ConstVid<'tcx>>>>,
{
    if let ty::ConstKind::Infer(InferConst::Var(vid)) = c.val() {
        match table.probe_value(vid).val.known() {
            Some(c) => c,
            None => c,
        }
    } else {
        c
    }
}
