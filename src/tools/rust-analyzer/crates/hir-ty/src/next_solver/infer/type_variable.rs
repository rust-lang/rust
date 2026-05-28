//! Storage for type variables for the infer context the next-trait-solver.

use std::cmp;
use std::marker::PhantomData;
use std::ops::Range;

use ena::snapshot_vec as sv;
use ena::undo_log::Rollback;
use ena::unify as ut;
use rustc_type_ir::TyVid;
use rustc_type_ir::UniverseIndex;
use rustc_type_ir::inherent::Ty as _;
use tracing::debug;

use crate::Span;
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

#[derive(Clone, Copy, Debug)]
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
            TypeVariableValue::Known { value, .. } => Some(*value),
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

    pub(super) fn finalize_rollback(&mut self) {}
}

impl<'db> TypeVariableTable<'_, 'db> {
    pub(crate) fn var_span(&mut self, vid: TyVid) -> Span {
        // We return the span from unification and not equation, since when equating we also unify,
        // and we want to prevent duplicate diagnostics from vars that were unified.
        self.sub_unification_table().probe_value(vid).span
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

    pub(crate) fn new_var(&mut self, universe: UniverseIndex, span: Span) -> TyVid {
        let eq_key = self.eq_relations().new_key(TypeVariableValue::Unknown { universe });

        let sub_key = self.sub_unification_table().new_key(TypeVariableSubValue { span });
        debug_assert_eq!(eq_key.vid, sub_key.vid);

        debug!("new_var(index={:?}, universe={:?}, span={:?})", eq_key.vid, universe, span);

        eq_key.vid
    }

    /// Returns the number of type variables created thus far.
    pub(crate) fn num_vars(&self) -> usize {
        self.storage.eq_relations.len()
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
    pub(crate) fn vars_since_snapshot(&mut self, value_count: usize) -> (Range<TyVid>, Vec<Span>) {
        let range = TyVid::from_usize(value_count)..TyVid::from_usize(self.num_vars());
        (range.clone(), iter_idx_range(range).map(|index| self.var_span(index)).collect())
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct TypeVariableSubValue {
    span: Span,
}

impl ut::UnifyKey for TyVidSubKey {
    type Value = TypeVariableSubValue;
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

impl ut::UnifyValue for TypeVariableSubValue {
    type Error = ut::NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        Ok(TypeVariableSubValue { span: Span::pick_best(value1.span, value2.span) })
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
            (&TypeVariableValue::Known { value }, &TypeVariableValue::Unknown { .. })
            | (&TypeVariableValue::Unknown { .. }, &TypeVariableValue::Known { value }) => {
                Ok(TypeVariableValue::Known { value })
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
