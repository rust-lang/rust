use ty::{self, FloatVarValue, IntVarValue, Ty, TyCtxt};
use rustc_data_structures::unify::{NoError, EqUnifyValue, UnifyKey, UnifyValue};

pub trait ToType {
    fn to_type<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>;
}

impl UnifyKey for ty::IntVid {
    type Value = Option<IntVarValue>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::IntVid { ty::IntVid { index: i } }
    fn tag() -> &'static str { "IntVid" }
}

impl EqUnifyValue for IntVarValue {
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct RegionVidKey {
    /// The minimum region vid in the unification set. This is needed
    /// to have a canonical name for a type to prevent infinite
    /// recursion.
    pub min_vid: ty::RegionVid
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
    fn index(&self) -> u32 { u32::from(*self) }
    fn from_index(i: u32) -> ty::RegionVid { ty::RegionVid::from(i) }
    fn tag() -> &'static str { "RegionVid" }
}

impl ToType for IntVarValue {
    fn to_type<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => tcx.mk_mach_int(i),
            ty::UintType(i) => tcx.mk_mach_uint(i),
        }
    }
}

// Floating point type keys

impl UnifyKey for ty::FloatVid {
    type Value = Option<FloatVarValue>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::FloatVid { ty::FloatVid { index: i } }
    fn tag() -> &'static str { "FloatVid" }
}

impl EqUnifyValue for FloatVarValue {
}

impl ToType for FloatVarValue {
    fn to_type<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        tcx.mk_mach_float(self.0)
    }
}
