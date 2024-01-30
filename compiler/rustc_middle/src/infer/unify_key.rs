use crate::ty::{self, Region, Ty, TyCtxt};
use rustc_data_structures::unify::{NoError, UnifyKey, UnifyValue};
use rustc_span::def_id::DefId;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use std::cmp;
use std::marker::PhantomData;

pub trait ToType {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct UnifiedRegion<'tcx> {
    value: Option<ty::Region<'tcx>>,
}

impl<'tcx> UnifiedRegion<'tcx> {
    pub fn new(value: Option<Region<'tcx>>) -> Self {
        Self { value }
    }

    /// The caller is responsible for checking universe compatibility before using this value.
    pub fn get_value_ignoring_universes(self) -> Option<Region<'tcx>> {
        self.value
    }
}

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
        // We pick the value of the least universe because it is compatible with more variables.
        // This is *not* necessary for completeness.
        #[cold]
        fn min_universe<'tcx>(r1: Region<'tcx>, r2: Region<'tcx>) -> Region<'tcx> {
            cmp::min_by_key(r1, r2, |r| match r.kind() {
                ty::ReStatic
                | ty::ReErased
                | ty::ReLateParam(..)
                | ty::ReEarlyParam(..)
                | ty::ReError(_) => ty::UniverseIndex::ROOT,
                ty::RePlaceholder(placeholder) => placeholder.universe,
                ty::ReVar(..) | ty::ReBound(..) => bug!("not a universal region"),
            })
        }

        Ok(match (value1.value, value2.value) {
            // Here we can just pick one value, because the full constraints graph
            // will be handled later. Ideally, we might want a `MultipleValues`
            // variant or something. For now though, this is fine.
            (Some(val1), Some(val2)) => Self { value: Some(min_universe(val1, val2)) },

            (Some(_), _) => *value1,
            (_, Some(_)) => *value2,

            (None, None) => *value1,
        })
    }
}

impl ToType for ty::IntVarValue {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => Ty::new_int(tcx, i),
            ty::UintType(i) => Ty::new_uint(tcx, i),
        }
    }
}

impl ToType for ty::FloatTy {
    fn to_type<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        Ty::new_float(tcx, *self)
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
}

#[derive(Copy, Clone, Debug)]
pub enum ConstVariableValue<'tcx> {
    Known { value: ty::Const<'tcx> },
    Unknown { origin: ConstVariableOrigin, universe: ty::UniverseIndex },
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
