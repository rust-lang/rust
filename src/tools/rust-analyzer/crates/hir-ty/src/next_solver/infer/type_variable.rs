//! Storage for type variables for the infer context the next-trait-solver.

use std::cmp;
use std::marker::PhantomData;
use std::ops::Range;

use ena::snapshot_vec as sv;
use ena::undo_log::Rollback;
use ena::unify as ut;
use rustc_index::IndexVec;
use rustc_type_ir::TyVid;
use rustc_type_ir::UniverseIndex;
use rustc_type_ir::inherent::Ty as _;
use tracing::debug;

use crate::next_solver::SolverDefId;
use crate::next_solver::Ty;
use crate::next_solver::infer::{InferCtxtUndoLogs, iter_idx_range};

/// Represents a single undo-able action that affects a type inference variable.
#[derive(Clone)]
pub(crate) enum UndoLog<'tcx> {
    EqRelation(sv::UndoLog<ut::Delegate<TyVidEqKey<'tcx>>>),
    SubRelation(sv::UndoLog<ut::Delegate<TyVidSubKey>>),
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'db> From<sv::UndoLog<ut::Delegate<TyVidEqKey<'db>>>> for UndoLog<'db> {
    fn from(l: sv::UndoLog<ut::Delegate<TyVidEqKey<'db>>>) -> Self {
        UndoLog::EqRelation(l)
    }
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'db> From<sv::UndoLog<ut::Delegate<TyVidSubKey>>> for UndoLog<'db> {
    fn from(l: sv::UndoLog<ut::Delegate<TyVidSubKey>>) -> Self {
        UndoLog::SubRelation(l)
    }
}

impl<'db> Rollback<sv::UndoLog<ut::Delegate<TyVidEqKey<'db>>>> for TypeVariableStorage<'db> {
    fn reverse(&mut self, undo: sv::UndoLog<ut::Delegate<TyVidEqKey<'db>>>) {
        self.eq_relations.reverse(undo)
    }
}

impl<'tcx> Rollback<sv::UndoLog<ut::Delegate<TyVidSubKey>>> for TypeVariableStorage<'tcx> {
    fn reverse(&mut self, undo: sv::UndoLog<ut::Delegate<TyVidSubKey>>) {
        self.sub_unification_table.reverse(undo)
    }
}

impl<'tcx> Rollback<UndoLog<'tcx>> for TypeVariableStorage<'tcx> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        match undo {
            UndoLog::EqRelation(undo) => self.eq_relations.reverse(undo),
            UndoLog::SubRelation(undo) => self.sub_unification_table.reverse(undo),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TypeVariableStorage<'db> {
    /// The origins of each type variable.
    values: IndexVec<TyVid, TypeVariableData>,
    /// Two variables are unified in `eq_relations` when we have a
    /// constraint `?X == ?Y`. This table also stores, for each key,
    /// the known value.
    eq_relations: ut::UnificationTableStorage<TyVidEqKey<'db>>,
    /// Only used by `-Znext-solver` and for diagnostics. Tracks whether
    /// type variables are related via subtyping at all, ignoring which of
    /// the two is the subtype.
    ///
    /// When reporting ambiguity errors, we sometimes want to
    /// treat all inference vars which are subtypes of each
    /// others as if they are equal. For this case we compute
    /// the transitive closure of our subtype obligations here.
    ///
    /// E.g. when encountering ambiguity errors, we want to suggest
    /// specifying some method argument or to add a type annotation
    /// to a local variable. Because subtyping cannot change the
    /// shape of a type, it's fine if the cause of the ambiguity error
    /// is only related to the suggested variable via subtyping.
    ///
    /// Even for something like `let x = returns_arg(); x.method();` the
    /// type of `x` is only a supertype of the argument of `returns_arg`. We
    /// still want to suggest specifying the type of the argument.
    sub_unification_table: ut::UnificationTableStorage<TyVidSubKey>,
}

pub(crate) struct TypeVariableTable<'a, 'db> {
    storage: &'a mut TypeVariableStorage<'db>,

    undo_log: &'a mut InferCtxtUndoLogs<'db>,
}

#[derive(Copy, Clone, Debug)]
pub struct TypeVariableOrigin {
    /// `DefId` of the type parameter this was instantiated for, if any.
    ///
    /// This should only be used for diagnostics.
    pub param_def_id: Option<SolverDefId>,
}

#[derive(Debug, Clone)]
pub(crate) struct TypeVariableData {
    origin: TypeVariableOrigin,
}

#[derive(Clone, Debug)]
pub(crate) enum TypeVariableValue<'db> {
    Known { value: Ty<'db> },
    Unknown { universe: UniverseIndex },
}

impl<'db> TypeVariableValue<'db> {
    /// If this value is known, returns the type it is known to be.
    /// Otherwise, `None`.
    pub(crate) fn known(&self) -> Option<Ty<'db>> {
        match self {
            TypeVariableValue::Unknown { .. } => None,
            TypeVariableValue::Known { value } => Some(*value),
        }
    }

    pub(crate) fn is_unknown(&self) -> bool {
        match *self {
            TypeVariableValue::Unknown { .. } => true,
            TypeVariableValue::Known { .. } => false,
        }
    }
}

impl<'db> TypeVariableStorage<'db> {
    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'db>,
    ) -> TypeVariableTable<'a, 'db> {
        TypeVariableTable { storage: self, undo_log }
    }

    #[inline]
    pub(crate) fn eq_relations_ref(&self) -> &ut::UnificationTableStorage<TyVidEqKey<'db>> {
        &self.eq_relations
    }

    pub(super) fn finalize_rollback(&mut self) {
        debug_assert!(self.values.len() >= self.eq_relations.len());
        self.values.truncate(self.eq_relations.len());
    }
}

impl<'db> TypeVariableTable<'_, 'db> {
    /// Returns the origin that was given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub(crate) fn var_origin(&self, vid: TyVid) -> TypeVariableOrigin {
        self.storage.values[vid].origin
    }

    /// Records that `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub(crate) fn equate(&mut self, a: TyVid, b: TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.eq_relations().union(a, b);
        self.sub_unification_table().union(a, b);
    }

    /// Records that `a` and `b` are related via subtyping. We don't track
    /// which of the two is the subtype.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub(crate) fn sub_unify(&mut self, a: TyVid, b: TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.sub_unification_table().union(a, b);
    }

    /// Instantiates `vid` with the type `ty`.
    ///
    /// Precondition: `vid` must not have been previously instantiated.
    pub(crate) fn instantiate(&mut self, vid: TyVid, ty: Ty<'db>) {
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
    pub(crate) fn new_var(&mut self, universe: UniverseIndex, origin: TypeVariableOrigin) -> TyVid {
        let eq_key = self.eq_relations().new_key(TypeVariableValue::Unknown { universe });

        let sub_key = self.sub_unification_table().new_key(());
        debug_assert_eq!(eq_key.vid, sub_key.vid);

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
    pub(crate) fn root_var(&mut self, vid: TyVid) -> TyVid {
        self.eq_relations().find(vid).vid
    }

    /// Returns the "root" variable of `vid` in the `sub_unification_table`
    /// equivalence table. All type variables that have been are related via
    /// equality or subtyping will yield the same root variable (per the
    /// union-find algorithm), so `sub_unification_table_root_var(a)
    /// == sub_unification_table_root_var(b)` implies that:
    /// ```text
    /// exists X. (a <: X || X <: a) && (b <: X || X <: b)
    /// ```
    pub(crate) fn sub_unification_table_root_var(&mut self, vid: TyVid) -> TyVid {
        self.sub_unification_table().find(vid).vid
    }

    /// Retrieves the type to which `vid` has been instantiated, if
    /// any.
    pub(crate) fn probe(&mut self, vid: TyVid) -> TypeVariableValue<'db> {
        self.inlined_probe(vid)
    }

    /// An always-inlined variant of `probe`, for very hot call sites.
    #[inline(always)]
    pub(crate) fn inlined_probe(&mut self, vid: TyVid) -> TypeVariableValue<'db> {
        self.eq_relations().inlined_probe_value(vid)
    }

    #[inline]
    fn eq_relations(&mut self) -> super::UnificationTable<'_, 'db, TyVidEqKey<'db>> {
        self.storage.eq_relations.with_log(self.undo_log)
    }

    #[inline]
    fn sub_unification_table(&mut self) -> super::UnificationTable<'_, 'db, TyVidSubKey> {
        self.storage.sub_unification_table.with_log(self.undo_log)
    }

    /// Returns a range of the type variables created during the snapshot.
    pub(crate) fn vars_since_snapshot(
        &mut self,
        value_count: usize,
    ) -> (Range<TyVid>, Vec<TypeVariableOrigin>) {
        let range = TyVid::from_usize(value_count)..TyVid::from_usize(self.num_vars());
        (range.clone(), iter_idx_range(range).map(|index| self.var_origin(index)).collect())
    }

    /// Returns indices of all variables that are not yet
    /// instantiated.
    pub(crate) fn unresolved_variables(&mut self) -> Vec<TyVid> {
        (0..self.num_vars())
            .filter_map(|i| {
                let vid = TyVid::from_usize(i);
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
pub(crate) struct TyVidEqKey<'db> {
    vid: TyVid,

    // in the table, we map each ty-vid to one of these:
    phantom: PhantomData<TypeVariableValue<'db>>,
}

impl<'db> From<TyVid> for TyVidEqKey<'db> {
    #[inline] // make this function eligible for inlining - it is quite hot.
    fn from(vid: TyVid) -> Self {
        TyVidEqKey { vid, phantom: PhantomData }
    }
}

impl<'db> ut::UnifyKey for TyVidEqKey<'db> {
    type Value = TypeVariableValue<'db>;
    #[inline(always)]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        TyVidEqKey::from(TyVid::from_u32(i))
    }
    fn tag() -> &'static str {
        "TyVidEqKey"
    }
    fn order_roots(a: Self, _: &Self::Value, b: Self, _: &Self::Value) -> Option<(Self, Self)> {
        if a.vid.as_u32() < b.vid.as_u32() { Some((a, b)) } else { Some((b, a)) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct TyVidSubKey {
    vid: TyVid,
}

impl From<TyVid> for TyVidSubKey {
    #[inline] // make this function eligible for inlining - it is quite hot.
    fn from(vid: TyVid) -> Self {
        TyVidSubKey { vid }
    }
}

impl ut::UnifyKey for TyVidSubKey {
    type Value = ();
    #[inline]
    fn index(&self) -> u32 {
        self.vid.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> TyVidSubKey {
        TyVidSubKey { vid: TyVid::from_u32(i) }
    }
    fn tag() -> &'static str {
        "TyVidSubKey"
    }
}

impl<'db> ut::UnifyValue for TypeVariableValue<'db> {
    type Error = ut::NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, ut::NoError> {
        match (value1, value2) {
            // We never equate two type variables, both of which
            // have known types. Instead, we recursively equate
            // those types.
            (&TypeVariableValue::Known { .. }, &TypeVariableValue::Known { .. }) => {
                panic!("equating two type variables, both of which have known types")
            }

            // If one side is known, prefer that one.
            (&TypeVariableValue::Known { .. }, &TypeVariableValue::Unknown { .. }) => {
                Ok(value1.clone())
            }
            (&TypeVariableValue::Unknown { .. }, &TypeVariableValue::Known { .. }) => {
                Ok(value2.clone())
            }

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
