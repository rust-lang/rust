use crate::ty::{self, FloatVarValue, InferConst, IntVarValue, Ty, TyCtxt};
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

/// Raw `TyVid` are used as the unification key for `sub_relations`;
/// they carry no values.
impl UnifyKey for ty::TyVid {
    type Value = ();
    fn index(&self) -> u32 {
        self.index
    }
    fn from_index(i: u32) -> ty::TyVid {
        ty::TyVid { index: i }
    }
    fn tag() -> &'static str {
        "TyVid"
    }
}

impl UnifyKey for ty::IntVid {
    type Value = Option<IntVarValue>;
    fn index(&self) -> u32 {
        self.index
    }
    fn from_index(i: u32) -> ty::IntVid {
        ty::IntVid { index: i }
    }
    fn tag() -> &'static str {
        "IntVid"
    }
}

impl EqUnifyValue for IntVarValue {}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct RegionVidKey {
    /// The minimum region vid in the unification set. This is needed
    /// to have a canonical name for a type to prevent infinite
    /// recursion.
    pub min_vid: ty::RegionVid,
}

impl UnifyValue for RegionVidKey {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, NoError> {
        let min_vid = if value1.min_vid.index() < value2.min_vid.index() {
            value1.min_vid
        } else {
            value2.min_vid
        };

        Ok(RegionVidKey { min_vid })
    }
}

impl UnifyKey for ty::RegionVid {
    type Value = RegionVidKey;
    fn index(&self) -> u32 {
        u32::from(*self)
    }
    fn from_index(i: u32) -> ty::RegionVid {
        ty::RegionVid::from(i)
    }
    fn tag() -> &'static str {
        "RegionVid"
    }
}

impl ToType for IntVarValue {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => tcx.mk_mach_int(i),
            ty::UintType(i) => tcx.mk_mach_uint(i),
        }
    }
}

// Floating point type keys

impl UnifyKey for ty::FloatVid {
    type Value = Option<FloatVarValue>;
    fn index(&self) -> u32 {
        self.index
    }
    fn from_index(i: u32) -> ty::FloatVid {
        ty::FloatVid { index: i }
    }
    fn tag() -> &'static str {
        "FloatVid"
    }
}

impl EqUnifyValue for FloatVarValue {}

impl ToType for FloatVarValue {
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
    Known { value: &'tcx ty::Const<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

impl<'tcx> ConstVariableValue<'tcx> {
    /// If this value is known, returns the const it is known to be.
    /// Otherwise, `None`.
    pub fn known(&self) -> Option<&'tcx ty::Const<'tcx>> {
        match *self {
            ConstVariableValue::Unknown { .. } => None,
            ConstVariableValue::Known { value } => Some(value),
        }
    }

    pub fn is_unknown(&self) -> bool {
        match *self {
            ConstVariableValue::Unknown { .. } => true,
            ConstVariableValue::Known { .. } => false,
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
    fn index(&self) -> u32 {
        self.index
    }
    fn from_index(i: u32) -> Self {
        ty::ConstVid { index: i, phantom: PhantomData }
    }
    fn tag() -> &'static str {
        "ConstVid"
    }
}

impl<'tcx> UnifyValue for ConstVarValue<'tcx> {
    type Error = (&'tcx ty::Const<'tcx>, &'tcx ty::Const<'tcx>);

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

impl<'tcx> EqUnifyValue for &'tcx ty::Const<'tcx> {}

pub fn replace_if_possible<V, L>(
    table: &mut UnificationTable<InPlace<ty::ConstVid<'tcx>, V, L>>,
    c: &'tcx ty::Const<'tcx>,
) -> &'tcx ty::Const<'tcx>
where
    V: snapshot_vec::VecLike<unify::Delegate<ty::ConstVid<'tcx>>>,
    L: UndoLogs<snapshot_vec::UndoLog<unify::Delegate<ty::ConstVid<'tcx>>>>,
{
    if let ty::Const { val: ty::ConstKind::Infer(InferConst::Var(vid)), .. } = c {
        match table.probe_value(*vid).val.known() {
            Some(c) => c,
            None => c,
        }
    } else {
        c
    }
}
