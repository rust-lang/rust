use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::ty::{self, Ty, TyVid};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

use crate::infer::InferCtxtUndoLogs;

use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;
use std::ops::Range;

use rustc_data_structures::undo_log::Rollback;

use super::unification_table::UndoLogDelegate;
use super::unification_table::UnificationStorage;
use super::unification_table::UnificationTable;
use super::unification_table::VariableValue;

/// Represents a single undo-able action that affects a type inference variable.
#[derive(Clone)]
pub(crate) enum UndoLog<'tcx> {
    EqRelation(UndoLogDelegate<'tcx, ty::TyVid>),
    SubRelation(sv::UndoLog<ut::Delegate<ty::TyVid>>),
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'tcx> From<UndoLogDelegate<'tcx, ty::TyVid>> for UndoLog<'tcx> {
    fn from(l: UndoLogDelegate<'tcx, ty::TyVid>) -> Self {
        UndoLog::EqRelation(l)
    }
}

/// Convert from a specific kind of undo to the more general UndoLog
impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::TyVid>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::TyVid>>) -> Self {
        UndoLog::SubRelation(l)
    }
}

impl<'tcx> Rollback<UndoLog<'tcx>> for TypeVariableStorage<'tcx> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        match undo {
            UndoLog::EqRelation(undo) => self.eq_relations.reverse(undo),
            UndoLog::SubRelation(undo) => self.sub_relations.reverse(undo),
        }
    }
}

#[derive(Clone)]
pub struct TypeVariableStorage<'tcx> {
    /// The origins of each type variable.
    values: IndexVec<TyVid, TypeVariableData>,
    /// Two variables are unified in `eq_relations` when we have a
    /// constraint `?X == ?Y`. This table also stores, for each key,
    /// the known value.
    eq_relations: UnificationStorage<'tcx, ty::TyVid>,

    /// Two variables are unified in `sub_relations` when we have a
    /// constraint `?X <: ?Y` *or* a constraint `?Y <: ?X`. This second
    /// table exists only to help with the occurs check. In particular,
    /// we want to report constraints like these as an occurs check
    /// violation:
    /// ``` text
    /// ?1 <: ?3
    /// Box<?3> <: ?1
    /// ```
    /// Without this second table, what would happen in a case like
    /// this is that we would instantiate `?1` with a generalized
    /// type like `Box<?6>`. We would then relate `Box<?3> <: Box<?6>`
    /// and infer that `?3 <: ?6`. Next, since `?1` was instantiated,
    /// we would process `?1 <: ?3`, generalize `?1 = Box<?6>` to `Box<?9>`,
    /// and instantiate `?3` with `Box<?9>`. Finally, we would relate
    /// `?6 <: ?9`. But now that we instantiated `?3`, we can process
    /// `?3 <: ?6`, which gives us `Box<?9> <: ?6`... and the cycle
    /// continues. (This is `occurs-check-2.rs`.)
    ///
    /// What prevents this cycle is that when we generalize
    /// `Box<?3>` to `Box<?6>`, we also sub-unify `?3` and `?6`
    /// (in the generalizer). When we then process `Box<?6> <: ?3`,
    /// the occurs check then fails because `?6` and `?3` are sub-unified,
    /// and hence generalization fails.
    ///
    /// This is reasonable because, in Rust, subtypes have the same
    /// "skeleton" and hence there is no possible type such that
    /// (e.g.)  `Box<?3> <: ?3` for any `?3`.
    ///
    /// In practice, we sometimes sub-unify variables in other spots, such
    /// as when processing subtype predicates. This is not necessary but is
    /// done to aid diagnostics, as it allows us to be more effective when
    /// we guide the user towards where they should insert type hints.
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
    OpaqueTypeInference(DefId),
    TypeParameterDefinition(Symbol, DefId),

    /// One of the upvars or closure kind parameters in a `ClosureArgs`
    /// (before it has been determined).
    // FIXME(eddyb) distinguish upvar inference variables from the rest.
    ClosureSynthetic,
    AutoDeref,
    AdjustmentType,

    /// In type check, when we are type checking a function that
    /// returns `-> dyn Foo`, we substitute a type variable for the
    /// return type for diagnostic purposes.
    DynReturnFn,
    LatticeVariable,
}

#[derive(Clone)]
pub(crate) struct TypeVariableData {
    origin: TypeVariableOrigin,
}

impl<'tcx> TypeVariableStorage<'tcx> {
    pub fn new() -> TypeVariableStorage<'tcx> {
        TypeVariableStorage {
            values: Default::default(),
            eq_relations: UnificationStorage::new(),
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

    #[inline]
    pub(crate) fn eq_relations_ref(&self) -> &UnificationStorage<'tcx, ty::TyVid> {
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
    pub fn var_origin(&self, vid: ty::TyVid) -> TypeVariableOrigin {
        self.storage.values[vid].origin
    }

    /// Records that `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn equate(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.eq_relations().unify(a, b);
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
        debug_assert!(!ty.is_ty_var(), "instantiating ty var with var: {vid:?} {ty:?}");
        debug_assert!(self.probe(vid).is_unknown());
        debug_assert!(
            self.eq_relations().probe_value(vid).is_unknown(),
            "instantiating type variable `{vid:?}` twice: new-value = {ty:?}, old-value={:?}",
            self.eq_relations().probe_value(vid)
        );
        self.eq_relations().instantiate(vid, ty);
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
        origin: TypeVariableOrigin,
    ) -> ty::TyVid {
        let eq_key = self.eq_relations().new_key(universe);

        let sub_key = self.sub_relations().new_key(());
        debug_assert_eq!(eq_key, sub_key);

        let index = self.storage.values.push(TypeVariableData { origin });
        debug_assert_eq!(eq_key, index);

        debug!("new_var(index={:?}, universe={:?}, origin={:?})", eq_key, universe, origin);

        index
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
        self.eq_relations().current_root(vid)
    }

    /// Returns the "root" variable of `vid` in the `sub_relations`
    /// equivalence table. All type variables that have been are
    /// related via equality or subtyping will yield the same root
    /// variable (per the union-find algorithm), so `sub_root_var(a)
    /// == sub_root_var(b)` implies that:
    /// ```text
    /// exists X. (a <: X || X <: a) && (b <: X || X <: b)
    /// ```
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
    pub fn probe(&mut self, vid: ty::TyVid) -> VariableValue<'tcx, ty::TyVid> {
        self.eq_relations().probe_value(vid)
    }

    /// An always-inlined variant of `probe`, for very hot call sites.
    #[inline(always)]
    pub fn inlined_probe(&mut self, vid: ty::TyVid) -> VariableValue<'tcx, ty::TyVid> {
        self.eq_relations().inlined_probe_value(vid)
    }

    #[inline]
    fn eq_relations(&mut self) -> UnificationTable<'_, 'tcx, ty::TyVid> {
        self.storage.eq_relations.with_log(self.undo_log)
    }

    #[inline]
    fn sub_relations(&mut self) -> super::RawUnificationTable<'_, 'tcx, ty::TyVid> {
        self.storage.sub_relations.with_log(self.undo_log)
    }

    /// Returns a range of the type variables created during the snapshot.
    pub fn vars_since_snapshot(
        &mut self,
        value_count: usize,
    ) -> (Range<TyVid>, Vec<TypeVariableOrigin>) {
        let range = TyVid::from_usize(value_count)..TyVid::from_usize(self.num_vars());
        (
            range.start..range.end,
            (range.start..range.end).map(|index| self.var_origin(index)).collect(),
        )
    }

    /// Returns indices of all variables that are not yet
    /// instantiated.
    pub fn unresolved_variables(&mut self) -> Vec<ty::TyVid> {
        (0..self.num_vars())
            .filter_map(|i| {
                let vid = ty::TyVid::from_usize(i);
                match self.probe(vid) {
                    VariableValue::Unknown { .. } => Some(vid),
                    VariableValue::Known { .. } => None,
                }
            })
            .collect()
    }
}
