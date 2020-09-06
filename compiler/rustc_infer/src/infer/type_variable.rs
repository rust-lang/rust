use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyVid};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

use crate::infer::InferCtxtUndoLogs;

use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;
use std::cmp;
use std::marker::PhantomData;
use std::ops::Range;

use rustc_data_structures::undo_log::{Rollback, UndoLogs};

/// Represents a single undo-able action that affects a type inference variable.
pub(crate) enum UndoLog<'tcx> {
    EqRelation(sv::UndoLog<ut::Delegate<TyVidEqKey<'tcx>>>),
    SubRelation(sv::UndoLog<ut::Delegate<ty::TyVid>>),
    Values(sv::UndoLog<Delegate>),
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'tcx> From<sv::UndoLog<ut::Delegate<TyVidEqKey<'tcx>>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<TyVidEqKey<'tcx>>>) -> Self {
        UndoLog::EqRelation(l)
    }
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::TyVid>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::TyVid>>) -> Self {
        UndoLog::SubRelation(l)
    }
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'tcx> From<sv::UndoLog<Delegate>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<Delegate>) -> Self {
        UndoLog::Values(l)
    }
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'tcx> From<Instantiate> for UndoLog<'tcx> {
    fn from(l: Instantiate) -> Self {
        UndoLog::Values(sv::UndoLog::Other(l))
    }
}

impl<'tcx> Rollback<UndoLog<'tcx>> for TypeVariableStorage<'tcx> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        match undo {
            UndoLog::EqRelation(undo) => self.eq_relations.reverse(undo),
            UndoLog::SubRelation(undo) => self.sub_relations.reverse(undo),
            UndoLog::Values(undo) => self.values.reverse(undo),
        }
    }
}

pub struct TypeVariableStorage<'tcx> {
    values: sv::SnapshotVecStorage<Delegate>,

    /// Two variables are unified in `eq_relations` when we have a
    /// constraint `?X == ?Y`. This table also stores, for each key,
    /// the known value.
    eq_relations: ut::UnificationTableStorage<TyVidEqKey<'tcx>>,

    /// Two variables are unified in `sub_relations` when we have a
    /// constraint `?X <: ?Y` *or* a constraint `?Y <: ?X`. This second
    /// table exists only to help with the occurs check. In particular,
    /// we want to report constraints like these as an occurs check
    /// violation:
    ///
    ///     ?1 <: ?3
    ///     Box<?3> <: ?1
    ///
    /// This works because `?1` and `?3` are unified in the
    /// `sub_relations` relation (not in `eq_relations`). Then when we
    /// process the `Box<?3> <: ?1` constraint, we do an occurs check
    /// on `Box<?3>` and find a potential cycle.
    ///
    /// This is reasonable because, in Rust, subtypes have the same
    /// "skeleton" and hence there is no possible type such that
    /// (e.g.)  `Box<?3> <: ?3` for any `?3`.
    sub_relations: ut::UnificationTableStorage<ty::TyVid>,
}

pub struct TypeVariableTable<'a, 'tcx> {
    storage: &'a mut TypeVariableStorage<'tcx>,

    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}

#[derive(Copy, Clone, Debug)]
pub struct TypeVariableOrigin {
    pub kind: TypeVariableOriginKind,
    pub span: Span,
}

/// Reasons to create a type inference variable
#[derive(Copy, Clone, Debug)]
pub enum TypeVariableOriginKind {
    MiscVariable,
    NormalizeProjectionType,
    TypeInference,
    TypeParameterDefinition(Symbol, Option<DefId>),

    /// One of the upvars or closure kind parameters in a `ClosureSubsts`
    /// (before it has been determined).
    // FIXME(eddyb) distinguish upvar inference variables from the rest.
    ClosureSynthetic,
    SubstitutionPlaceholder,
    AutoDeref,
    AdjustmentType,
    DivergingFn,
    LatticeVariable,
}

pub(crate) struct TypeVariableData {
    origin: TypeVariableOrigin,
    diverging: bool,
}

#[derive(Copy, Clone, Debug)]
pub enum TypeVariableValue<'tcx> {
    Known { value: Ty<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

impl<'tcx> TypeVariableValue<'tcx> {
    /// If this value is known, returns the type it is known to be.
    /// Otherwise, `None`.
    pub fn known(&self) -> Option<Ty<'tcx>> {
        match *self {
            TypeVariableValue::Unknown { .. } => None,
            TypeVariableValue::Known { value } => Some(value),
        }
    }

    pub fn is_unknown(&self) -> bool {
        match *self {
            TypeVariableValue::Unknown { .. } => true,
            TypeVariableValue::Known { .. } => false,
        }
    }
}

pub(crate) struct Instantiate {
    vid: ty::TyVid,
}

pub(crate) struct Delegate;

impl<'tcx> TypeVariableStorage<'tcx> {
    pub fn new() -> TypeVariableStorage<'tcx> {
        TypeVariableStorage {
            values: sv::SnapshotVecStorage::new(),
            eq_relations: ut::UnificationTableStorage::new(),
            sub_relations: ut::UnificationTableStorage::new(),
        }
    }

    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> TypeVariableTable<'a, 'tcx> {
        TypeVariableTable { storage: self, undo_log }
    }
}

impl<'tcx> TypeVariableTable<'_, 'tcx> {
    /// Returns the diverges flag given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub fn var_diverges(&self, vid: ty::TyVid) -> bool {
        self.storage.values.get(vid.index as usize).diverging
    }

    /// Returns the origin that was given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub fn var_origin(&self, vid: ty::TyVid) -> &TypeVariableOrigin {
        &self.storage.values.get(vid.index as usize).origin
    }

    /// Records that `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn equate(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.eq_relations().union(a, b);
        self.sub_relations().union(a, b);
    }

    /// Records that `a <: b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn sub(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.sub_relations().union(a, b);
    }

    /// Instantiates `vid` with the type `ty`.
    ///
    /// Precondition: `vid` must not have been previously instantiated.
    pub fn instantiate(&mut self, vid: ty::TyVid, ty: Ty<'tcx>) {
        let vid = self.root_var(vid);
        debug_assert!(self.probe(vid).is_unknown());
        debug_assert!(
            self.eq_relations().probe_value(vid).is_unknown(),
            "instantiating type variable `{:?}` twice: new-value = {:?}, old-value={:?}",
            vid,
            ty,
            self.eq_relations().probe_value(vid)
        );
        self.eq_relations().union_value(vid, TypeVariableValue::Known { value: ty });

        // Hack: we only need this so that `types_escaping_snapshot`
        // can see what has been unified; see the Delegate impl for
        // more details.
        self.undo_log.push(Instantiate { vid });
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
    pub fn new_var(
        &mut self,
        universe: ty::UniverseIndex,
        diverging: bool,
        origin: TypeVariableOrigin,
    ) -> ty::TyVid {
        let eq_key = self.eq_relations().new_key(TypeVariableValue::Unknown { universe });

        let sub_key = self.sub_relations().new_key(());
        assert_eq!(eq_key.vid, sub_key);

        let index = self.values().push(TypeVariableData { origin, diverging });
        assert_eq!(eq_key.vid.index, index as u32);

        debug!(
            "new_var(index={:?}, universe={:?}, diverging={:?}, origin={:?}",
            eq_key.vid, universe, diverging, origin,
        );

        eq_key.vid
    }

    /// Returns the number of type variables created thus far.
    pub fn num_vars(&self) -> usize {
        self.storage.values.len()
    }

    /// Returns the "root" variable of `vid` in the `eq_relations`
    /// equivalence table. All type variables that have been equated
    /// will yield the same root variable (per the union-find
    /// algorithm), so `root_var(a) == root_var(b)` implies that `a ==
    /// b` (transitively).
    pub fn root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.eq_relations().find(vid).vid
    }

    /// Returns the "root" variable of `vid` in the `sub_relations`
    /// equivalence table. All type variables that have been are
    /// related via equality or subtyping will yield the same root
    /// variable (per the union-find algorithm), so `sub_root_var(a)
    /// == sub_root_var(b)` implies that:
    ///
    ///     exists X. (a <: X || X <: a) && (b <: X || X <: b)
    pub fn sub_root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.sub_relations().find(vid)
    }

    /// Returns `true` if `a` and `b` have same "sub-root" (i.e., exists some
    /// type X such that `forall i in {a, b}. (i <: X || X <: i)`.
    pub fn sub_unified(&mut self, a: ty::TyVid, b: ty::TyVid) -> bool {
        self.sub_root_var(a) == self.sub_root_var(b)
    }

    /// Retrieves the type to which `vid` has been instantiated, if
    /// any.
    pub fn probe(&mut self, vid: ty::TyVid) -> TypeVariableValue<'tcx> {
        self.inlined_probe(vid)
    }

    /// An always-inlined variant of `probe`, for very hot call sites.
    #[inline(always)]
    pub fn inlined_probe(&mut self, vid: ty::TyVid) -> TypeVariableValue<'tcx> {
        self.eq_relations().inlined_probe_value(vid)
    }

    /// If `t` is a type-inference variable, and it has been
    /// instantiated, then return the with which it was
    /// instantiated. Otherwise, returns `t`.
    pub fn replace_if_possible(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Infer(ty::TyVar(v)) => match self.probe(v) {
                TypeVariableValue::Unknown { .. } => t,
                TypeVariableValue::Known { value } => value,
            },
            _ => t,
        }
    }

    #[inline]
    fn values(
        &mut self,
    ) -> sv::SnapshotVec<Delegate, &mut Vec<TypeVariableData>, &mut InferCtxtUndoLogs<'tcx>> {
        self.storage.values.with_log(self.undo_log)
    }

    #[inline]
    fn eq_relations(&mut self) -> super::UnificationTable<'_, 'tcx, TyVidEqKey<'tcx>> {
        self.storage.eq_relations.with_log(self.undo_log)
    }

    #[inline]
    fn sub_relations(&mut self) -> super::UnificationTable<'_, 'tcx, ty::TyVid> {
        self.storage.sub_relations.with_log(self.undo_log)
    }

    /// Returns a range of the type variables created during the snapshot.
    pub fn vars_since_snapshot(
        &mut self,
        value_count: usize,
    ) -> (Range<TyVid>, Vec<TypeVariableOrigin>) {
        let range = TyVid { index: value_count as u32 }..TyVid { index: self.num_vars() as u32 };
        (
            range.start..range.end,
            (range.start.index..range.end.index)
                .map(|index| self.storage.values.get(index as usize).origin)
                .collect(),
        )
    }

    /// Finds the set of type variables that existed *before* `s`
    /// but which have only been unified since `s` started, and
    /// return the types with which they were unified. So if we had
    /// a type variable `V0`, then we started the snapshot, then we
    /// created a type variable `V1`, unified `V0` with `T0`, and
    /// unified `V1` with `T1`, this function would return `{T0}`.
    pub fn types_escaping_snapshot(&mut self, s: &super::Snapshot<'tcx>) -> Vec<Ty<'tcx>> {
        let mut new_elem_threshold = u32::MAX;
        let mut escaping_types = Vec::new();
        let actions_since_snapshot = self.undo_log.actions_since_snapshot(s);
        debug!("actions_since_snapshot.len() = {}", actions_since_snapshot.len());
        for i in 0..actions_since_snapshot.len() {
            let actions_since_snapshot = self.undo_log.actions_since_snapshot(s);
            match actions_since_snapshot[i] {
                super::UndoLog::TypeVariables(UndoLog::Values(sv::UndoLog::NewElem(index))) => {
                    // if any new variables were created during the
                    // snapshot, remember the lower index (which will
                    // always be the first one we see). Note that this
                    // action must precede those variables being
                    // specified.
                    new_elem_threshold = cmp::min(new_elem_threshold, index as u32);
                    debug!("NewElem({}) new_elem_threshold={}", index, new_elem_threshold);
                }

                super::UndoLog::TypeVariables(UndoLog::Values(sv::UndoLog::Other(
                    Instantiate { vid, .. },
                ))) => {
                    if vid.index < new_elem_threshold {
                        // quick check to see if this variable was
                        // created since the snapshot started or not.
                        let mut eq_relations = ut::UnificationTable::with_log(
                            &mut self.storage.eq_relations,
                            &mut *self.undo_log,
                        );
                        let escaping_type = match eq_relations.probe_value(vid) {
                            TypeVariableValue::Unknown { .. } => bug!(),
                            TypeVariableValue::Known { value } => value,
                        };
                        escaping_types.push(escaping_type);
                    }
                    debug!("SpecifyVar({:?}) new_elem_threshold={}", vid, new_elem_threshold);
                }

                _ => {}
            }
        }

        escaping_types
    }

    /// Returns indices of all variables that are not yet
    /// instantiated.
    pub fn unsolved_variables(&mut self) -> Vec<ty::TyVid> {
        (0..self.storage.values.len())
            .filter_map(|i| {
                let vid = ty::TyVid { index: i as u32 };
                match self.probe(vid) {
                    TypeVariableValue::Unknown { .. } => Some(vid),
                    TypeVariableValue::Known { .. } => None,
                }
            })
            .collect()
    }
}

impl sv::SnapshotVecDelegate for Delegate {
    type Value = TypeVariableData;
    type Undo = Instantiate;

    fn reverse(_values: &mut Vec<TypeVariableData>, _action: Instantiate) {
        // We don't actually have to *do* anything to reverse an
        // instantiation; the value for a variable is stored in the
        // `eq_relations` and hence its rollback code will handle
        // it. In fact, we could *almost* just remove the
        // `SnapshotVec` entirely, except that we would have to
        // reproduce *some* of its logic, since we want to know which
        // type variables have been instantiated since the snapshot
        // was started, so we can implement `types_escaping_snapshot`.
        //
        // (If we extended the `UnificationTable` to let us see which
        // values have been unified and so forth, that might also
        // suffice.)
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
    fn from(vid: ty::TyVid) -> Self {
        TyVidEqKey { vid, phantom: PhantomData }
    }
}

impl<'tcx> ut::UnifyKey for TyVidEqKey<'tcx> {
    type Value = TypeVariableValue<'tcx>;
    fn index(&self) -> u32 {
        self.vid.index
    }
    fn from_index(i: u32) -> Self {
        TyVidEqKey::from(ty::TyVid { index: i })
    }
    fn tag() -> &'static str {
        "TyVidEqKey"
    }
}

impl<'tcx> ut::UnifyValue for TypeVariableValue<'tcx> {
    type Error = ut::NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, ut::NoError> {
        match (value1, value2) {
            // We never equate two type variables, both of which
            // have known types.  Instead, we recursively equate
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
