use std::cmp;
use std::marker::PhantomData;
use std::ops::Range;

use rustc_data_structures::undo_log::Rollback;
use rustc_data_structures::{snapshot_vec as sv, unify as ut};
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::ty::{self, Ty, TyVid};
use rustc_span::Span;
use tracing::debug;

use crate::infer::InferCtxtUndoLogs;

impl<'tcx> Rollback<sv::UndoLog<ut::Delegate<TyVidEqKey<'tcx>>>> for TypeVariableStorage<'tcx> {
    fn reverse(&mut self, undo: sv::UndoLog<ut::Delegate<TyVidEqKey<'tcx>>>) {
        self.eq_relations.reverse(undo)
    }
}

#[derive(Clone, Default)]
pub(crate) struct TypeVariableStorage<'tcx> {
    /// The origins of each type variable.
    values: IndexVec<TyVid, TypeVariableData>,
    /// Two variables are unified in `eq_relations` when we have a
    /// constraint `?X == ?Y`. This table also stores, for each key,
    /// the known value.
    eq_relations: ut::UnificationTableStorage<TyVidEqKey<'tcx>>,
}

pub(crate) struct TypeVariableTable<'a, 'tcx> {
    storage: &'a mut TypeVariableStorage<'tcx>,

    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}

#[derive(Copy, Clone, Debug)]
pub struct TypeVariableOrigin {
    pub span: Span,
    /// `DefId` of the type parameter this was instantiated for, if any.
    ///
    /// This should only be used for diagnostics.
    pub param_def_id: Option<DefId>,
}

#[derive(Clone)]
pub(crate) struct TypeVariableData {
    origin: TypeVariableOrigin,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum TypeVariableValue<'tcx> {
    Known { value: Ty<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

impl<'tcx> TypeVariableValue<'tcx> {
    /// If this value is known, returns the type it is known to be.
    /// Otherwise, `None`.
    pub(crate) fn known(&self) -> Option<Ty<'tcx>> {
        match *self {
            TypeVariableValue::Unknown { .. } => None,
            TypeVariableValue::Known { value } => Some(value),
        }
    }

    pub(crate) fn is_unknown(&self) -> bool {
        match *self {
            TypeVariableValue::Unknown { .. } => true,
            TypeVariableValue::Known { .. } => false,
        }
    }
}

impl<'tcx> TypeVariableStorage<'tcx> {
    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> TypeVariableTable<'a, 'tcx> {
        TypeVariableTable { storage: self, undo_log }
    }

    #[inline]
    pub(crate) fn eq_relations_ref(&self) -> &ut::UnificationTableStorage<TyVidEqKey<'tcx>> {
        &self.eq_relations
    }

    pub(super) fn finalize_rollback(&mut self) {
        debug_assert!(self.values.len() >= self.eq_relations.len());
        self.values.truncate(self.eq_relations.len());
    }
}

impl<'tcx> TypeVariableTable<'_, 'tcx> {
    /// Returns the origin that was given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub(crate) fn var_origin(&self, vid: ty::TyVid) -> TypeVariableOrigin {
        self.storage.values[vid].origin
    }

    /// Records that `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub(crate) fn equate(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.eq_relations().union(a, b);
    }

    /// Instantiates `vid` with the type `ty`.
    ///
    /// Precondition: `vid` must not have been previously instantiated.
    pub(crate) fn instantiate(&mut self, vid: ty::TyVid, ty: Ty<'tcx>) {
        let vid = self.root_var(vid);
        debug_assert!(!ty.is_ty_var(), "instantiating ty var with var: {vid:?} {ty:?}");
        debug_assert!(self.probe(vid).is_unknown());
        debug_assert!(
            self.eq_relations().probe_value(vid).is_unknown(),
            "instantiating type variable `{vid:?}` twice: new-value = {ty:?}, old-value={:?}",
            self.eq_relations().probe_value(vid)
        );
        self.eq_relations().union_value(vid, TypeVariableValue::Known { value: ty });
    }

    /// Creates a new type variable.
    ///
    /// - `diverging`: indicates if this is a "diverging" type
    ///   variable, e.g.,  one created as the type of a `return`
    ///   expression. The code in this module doesn't care if a
    ///   variable is diverging, but the main Rust type-checker will
    ///   sometimes "unify" such variables with the `!` or `()` types.
    /// - `origin`: indicates *why* the type variable was created.
    ///   The code in this module doesn't care, but it can be useful
    ///   for improving error messages.
    pub(crate) fn new_var(
        &mut self,
        universe: ty::UniverseIndex,
        origin: TypeVariableOrigin,
    ) -> ty::TyVid {
        let eq_key = self.eq_relations().new_key(TypeVariableValue::Unknown { universe });
        let index = self.storage.values.push(TypeVariableData { origin });
        debug_assert_eq!(eq_key.vid, index);

        debug!("new_var(index={:?}, universe={:?}, origin={:?})", eq_key.vid, universe, origin);

        index
    }

    /// Returns the number of type variables created thus far.
    pub(crate) fn num_vars(&self) -> usize {
        self.storage.values.len()
    }

    /// Returns the "root" variable of `vid` in the `eq_relations`
    /// equivalence table. All type variables that have been equated
    /// will yield the same root variable (per the union-find
    /// algorithm), so `root_var(a) == root_var(b)` implies that `a ==
    /// b` (transitively).
    pub(crate) fn root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.eq_relations().find(vid).vid
    }

    /// Retrieves the type to which `vid` has been instantiated, if
    /// any.
    pub(crate) fn probe(&mut self, vid: ty::TyVid) -> TypeVariableValue<'tcx> {
        self.inlined_probe(vid)
    }

    /// An always-inlined variant of `probe`, for very hot call sites.
    #[inline(always)]
    pub(crate) fn inlined_probe(&mut self, vid: ty::TyVid) -> TypeVariableValue<'tcx> {
        self.eq_relations().inlined_probe_value(vid)
    }

    #[inline]
    fn eq_relations(&mut self) -> super::UnificationTable<'_, 'tcx, TyVidEqKey<'tcx>> {
        self.storage.eq_relations.with_log(self.undo_log)
    }

    /// Returns a range of the type variables created during the snapshot.
    pub(crate) fn vars_since_snapshot(
        &mut self,
        value_count: usize,
    ) -> (Range<TyVid>, Vec<TypeVariableOrigin>) {
        let range = TyVid::from_usize(value_count)..TyVid::from_usize(self.num_vars());
        (range.clone(), range.map(|index| self.var_origin(index)).collect())
    }

    /// Returns indices of all variables that are not yet
    /// instantiated.
    pub(crate) fn unresolved_variables(&mut self) -> Vec<ty::TyVid> {
        (0..self.num_vars())
            .filter_map(|i| {
                let vid = ty::TyVid::from_usize(i);
                match self.probe(vid) {
                    TypeVariableValue::Unknown { .. } => Some(vid),
                    TypeVariableValue::Known { .. } => None,
                }
            })
            .collect()
    }
}

///////////////////////////////////////////////////////////////////////////

/// These structs (a newtyped TyVid) are used as the unification key
/// for the `eq_relations`; they carry a `TypeVariableValue` along
/// with them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct TyVidEqKey<'tcx> {
    vid: ty::TyVid,

    // in the table, we map each ty-vid to one of these:
    phantom: PhantomData<TypeVariableValue<'tcx>>,
}

impl<'tcx> From<ty::TyVid> for TyVidEqKey<'tcx> {
    #[inline] // make this function eligible for inlining - it is quite hot.
    fn from(vid: ty::TyVid) -> Self {
        TyVidEqKey { vid, phantom: PhantomData }
    }
}

impl<'tcx> ut::UnifyKey for TyVidEqKey<'tcx> {
    type Value = TypeVariableValue<'tcx>;
    #[inline(always)]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        TyVidEqKey::from(ty::TyVid::from_u32(i))
    }
    fn tag() -> &'static str {
        "TyVidEqKey"
    }
    fn order_roots(a: Self, _: &Self::Value, b: Self, _: &Self::Value) -> Option<(Self, Self)> {
        if a.vid.as_u32() < b.vid.as_u32() { Some((a, b)) } else { Some((b, a)) }
    }
}

impl<'tcx> ut::UnifyValue for TypeVariableValue<'tcx> {
    type Error = ut::NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, ut::NoError> {
        match (value1, value2) {
            // We never equate two type variables, both of which
            // have known types. Instead, we recursively equate
            // those types.
            (&TypeVariableValue::Known { .. }, &TypeVariableValue::Known { .. }) => {
                bug!("equating two type variables, both of which have known types")
            }

            // If one side is known, prefer that one.
            (&TypeVariableValue::Known { .. }, &TypeVariableValue::Unknown { .. }) => Ok(*value1),
            (&TypeVariableValue::Unknown { .. }, &TypeVariableValue::Known { .. }) => Ok(*value2),

            // If both sides are *unknown*, it hardly matters, does it?
            (
                &TypeVariableValue::Unknown { universe: universe1 },
                &TypeVariableValue::Unknown { universe: universe2 },
            ) => {
                // If we unify two unbound variables, ?T and ?U, then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                let universe = cmp::min(universe1, universe2);
                Ok(TypeVariableValue::Unknown { universe })
            }
        }
    }
}
