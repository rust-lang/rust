// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax_pos::Span;
use ty::{self, Ty};

use std::cmp;
use std::marker::PhantomData;
use std::u32;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::unify as ut;

pub struct TypeVariableTable<'tcx> {
    /// Extra data for each type variable, such as the origin. This is
    /// not stored in the unification table since, when we inquire
    /// after the origin of a variable X, we want the origin of **that
    /// variable X**, not the origin of some other variable Y with
    /// which X has been unified.
    var_data: Vec<TypeVariableData>,

    /// Two variables are unified in `eq_relations` when we have a
    /// constraint `?X == ?Y`. This table also stores, for each key,
    /// the known value.
    eq_relations: ut::UnificationTable<ut::InPlace<TyVidEqKey<'tcx>>>,

    /// Two variables are unified in `eq_relations` when we have a
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
    sub_relations: ut::UnificationTable<ut::InPlace<ty::TyVid>>,
}

/// Reasons to create a type inference variable
#[derive(Copy, Clone, Debug)]
pub enum TypeVariableOrigin {
    MiscVariable(Span),
    NormalizeProjectionType(Span),
    TypeInference(Span),
    TypeParameterDefinition(Span, ast::Name),

    /// one of the upvars or closure kind parameters in a `ClosureSubsts`
    /// (before it has been determined)
    ClosureSynthetic(Span),
    SubstitutionPlaceholder(Span),
    AutoDeref(Span),
    AdjustmentType(Span),
    DivergingStmt(Span),
    DivergingBlockExpr(Span),
    DivergingFn(Span),
    LatticeVariable(Span),
    Generalized(ty::TyVid),
}

pub type TypeVariableMap = FxHashMap<ty::TyVid, TypeVariableOrigin>;

struct TypeVariableData {
    origin: TypeVariableOrigin,
    diverging: bool
}

#[derive(Copy, Clone, Debug)]
pub enum TypeVariableValue<'tcx> {
    Known { value: Ty<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

#[derive(Copy, Clone, Debug)]
pub enum ProbeTyValue<'tcx> {
    Ty(Ty<'tcx>),
    Vid(ty::TyVid),
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

    /// If this value is unknown, returns the universe, otherwise `None`.
    pub fn universe(&self) -> Option<ty::UniverseIndex> {
        match *self {
            TypeVariableValue::Unknown { universe } => Some(universe),
            TypeVariableValue::Known { .. } => None,
        }
    }

    pub fn is_unknown(&self) -> bool {
        match *self {
            TypeVariableValue::Unknown { .. } => true,
            TypeVariableValue::Known { .. } => false,
        }
    }
}

pub struct Snapshot<'tcx> {
    /// number of variables at the time of the snapshot
    num_vars: usize,

    /// snapshot from the `eq_relations` table
    eq_snapshot: ut::Snapshot<ut::InPlace<TyVidEqKey<'tcx>>>,

    /// snapshot from the `sub_relations` table
    sub_snapshot: ut::Snapshot<ut::InPlace<ty::TyVid>>,
}

impl<'tcx> TypeVariableTable<'tcx> {
    pub fn new() -> TypeVariableTable<'tcx> {
        TypeVariableTable {
            var_data: Vec::new(),
            eq_relations: ut::UnificationTable::new(),
            sub_relations: ut::UnificationTable::new(),
        }
    }

    /// Returns the diverges flag given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub fn var_diverges<'a>(&'a self, vid: ty::TyVid) -> bool {
        self.var_data[vid.index as usize].diverging
    }

    /// Returns the origin that was given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub fn var_origin(&self, vid: ty::TyVid) -> &TypeVariableOrigin {
        &self.var_data[vid.index as usize].origin
    }

    /// Records that `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn equate(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.eq_relations.union(a, b);
        self.sub_relations.union(a, b);
    }

    /// Records that `a <: b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn sub(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_unknown());
        debug_assert!(self.probe(b).is_unknown());
        self.sub_relations.union(a, b);
    }

    /// Instantiates `vid` with the type `ty`.
    ///
    /// Precondition: `vid` must not have been previously instantiated.
    pub fn instantiate(&mut self, vid: ty::TyVid, ty: Ty<'tcx>) {
        let vid = self.root_var(vid);
        debug_assert!(self.probe(vid).is_unknown());
        debug_assert!(self.eq_relations.probe_value(vid).is_unknown(),
                      "instantiating type variable `{:?}` twice: new-value = {:?}, old-value={:?}",
                      vid, ty, self.eq_relations.probe_value(vid));
        self.eq_relations.union_value(vid, TypeVariableValue::Known { value: ty });
    }

    /// Creates a new type variable.
    ///
    /// - `diverging`: indicates if this is a "diverging" type
    ///   variable, e.g.  one created as the type of a `return`
    ///   expression. The code in this module doesn't care if a
    ///   variable is diverging, but the main Rust type-checker will
    ///   sometimes "unify" such variables with the `!` or `()` types.
    /// - `origin`: indicates *why* the type variable was created.
    ///   The code in this module doesn't care, but it can be useful
    ///   for improving error messages.
    pub fn new_var(&mut self,
                   universe: ty::UniverseIndex,
                   diverging: bool,
                   origin: TypeVariableOrigin)
                   -> ty::TyVid {
        let eq_key = self.eq_relations.new_key(TypeVariableValue::Unknown { universe });

        let sub_key = self.sub_relations.new_key(());
        assert_eq!(eq_key.vid, sub_key);

        assert_eq!(self.var_data.len(), sub_key.index as usize);
        self.var_data.push(TypeVariableData { origin, diverging });

        debug!("new_var(index={:?}, diverging={:?}, origin={:?}", eq_key.vid, diverging, origin);

        eq_key.vid
    }

    /// Returns the number of type variables created thus far.
    pub fn num_vars(&self) -> usize {
        self.var_data.len()
    }

    /// Returns the "root" variable of `vid` in the `eq_relations`
    /// equivalence table. All type variables that have been equated
    /// will yield the same root variable (per the union-find
    /// algorithm), so `root_var(a) == root_var(b)` implies that `a ==
    /// b` (transitively).
    pub fn root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.eq_relations.find(vid).vid
    }

    /// Returns the "root" variable of `vid` in the `sub_relations`
    /// equivalence table. All type variables that have been are
    /// related via equality or subtyping will yield the same root
    /// variable (per the union-find algorithm), so `sub_root_var(a)
    /// == sub_root_var(b)` implies that:
    ///
    ///     exists X. (a <: X || X <: a) && (b <: X || X <: b)
    pub fn sub_root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.sub_relations.find(vid)
    }

    /// True if `a` and `b` have same "sub-root" (i.e., exists some
    /// type X such that `forall i in {a, b}. (i <: X || X <: i)`.
    pub fn sub_unified(&mut self, a: ty::TyVid, b: ty::TyVid) -> bool {
        self.sub_root_var(a) == self.sub_root_var(b)
    }

    /// Retrieves the type to which `vid` has been instantiated, if
    /// any.
    pub fn probe(&mut self, vid: ty::TyVid) -> TypeVariableValue<'tcx> {
        self.eq_relations.probe_value(vid)
    }

    /// If `t` is a type-inference variable, and it has been
    /// instantiated, then return the with which it was
    /// instantiated. Otherwise, returns `t`.
    pub fn replace_if_possible(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::TyInfer(ty::TyVar(v)) => {
                match self.probe(v) {
                    TypeVariableValue::Unknown { .. } => t,
                    TypeVariableValue::Known { value } => value,
                }
            }
            _ => t,
        }
    }

    /// Creates a snapshot of the type variable state.  This snapshot
    /// must later be committed (`commit()`) or rolled back
    /// (`rollback_to()`).  Nested snapshots are permitted, but must
    /// be processed in a stack-like fashion.
    pub fn snapshot(&mut self) -> Snapshot<'tcx> {
        Snapshot {
            num_vars: self.var_data.len(),
            eq_snapshot: self.eq_relations.snapshot(),
            sub_snapshot: self.sub_relations.snapshot(),
        }
    }

    /// Undoes all changes since the snapshot was created. Any
    /// snapshots created since that point must already have been
    /// committed or rolled back.
    pub fn rollback_to(&mut self, s: Snapshot<'tcx>) {
        let Snapshot { num_vars, eq_snapshot, sub_snapshot } = s;
        debug!("type_variables::rollback_to(num_vars = {})", num_vars);
        assert!(self.var_data.len() >= num_vars);
        self.eq_relations.rollback_to(eq_snapshot);
        self.sub_relations.rollback_to(sub_snapshot);
        self.var_data.truncate(num_vars);
    }

    /// Commits all changes since the snapshot was created, making
    /// them permanent (unless this snapshot was created within
    /// another snapshot). Any snapshots created since that point
    /// must already have been committed or rolled back.
    pub fn commit(&mut self, s: Snapshot<'tcx>) {
        let Snapshot { num_vars, eq_snapshot, sub_snapshot } = s;
        debug!("type_variables::commit(num_vars = {})", num_vars);
        self.eq_relations.commit(eq_snapshot);
        self.sub_relations.commit(sub_snapshot);
    }

    /// Returns a map `{V1 -> V2}`, where the keys `{V1}` are
    /// ty-variables created during the snapshot, and the values
    /// `{V2}` are the root variables that they were unified with,
    /// along with their origin.
    pub fn types_created_since_snapshot(&mut self, snapshot: &Snapshot<'tcx>) -> TypeVariableMap {
        self.var_data
            .iter()
            .enumerate()
            .skip(snapshot.num_vars) // skip those that existed when snapshot was taken
            .map(|(index, data)| (ty::TyVid { index: index as u32 }, data.origin))
            .collect()
    }

    /// Find the set of type variables that existed *before* `s`
    /// but which have only been unified since `s` started, and
    /// return the types with which they were unified. So if we had
    /// a type variable `V0`, then we started the snapshot, then we
    /// created a type variable `V1`, unifed `V0` with `T0`, and
    /// unified `V1` with `T1`, this function would return `{T0}`.
    pub fn types_escaping_snapshot(&mut self, snapshot: &Snapshot<'tcx>) -> Vec<Ty<'tcx>> {
        // We want to select only those instantiations that have
        // occurred since the snapshot *and* which affect some
        // variable that existed prior to the snapshot. This code just
        // affects all instantiatons that ever occurred which affect
        // variables prior to the snapshot.
        //
        // It's hard to do better than this, though, without changing
        // the unification table to prefer "lower" vids -- the problem
        // is that we may have a variable X (from before the snapshot)
        // and Y (from after the snapshot) which get unified, with Y
        // chosen as the new root. Now we are "instantiating" Y with a
        // value, but it escapes into X, but we wouldn't readily see
        // that. (In fact, earlier revisions of this code had this
        // bug; it was introduced when we added the `eq_relations`
        // table, but it's hard to create rust code that triggers it.)
        //
        // We could tell the table to prefer lower vids, and then we would
        // see the case above, but we would get less-well-balanced trees.
        //
        // Since I hope to kill the leak-check in this branch, and
        // that's the code which uses this logic anyway, I'm going to
        // use the less efficient algorithm for now.
        let mut escaping_types = Vec::with_capacity(snapshot.num_vars);
        escaping_types.extend(
            (0..snapshot.num_vars) // for all variables that pre-exist the snapshot, collect..
                .map(|i| ty::TyVid { index: i as u32 })
                .filter_map(|vid| self.probe(vid).known())); // ..types they are instantiated with.
        debug!("types_escaping_snapshot = {:?}", escaping_types);
        escaping_types
    }

    /// Returns indices of all variables that are not yet
    /// instantiated.
    pub fn unsolved_variables(&mut self) -> Vec<ty::TyVid> {
        (0..self.var_data.len())
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
///////////////////////////////////////////////////////////////////////////

/// These structs (a newtyped TyVid) are used as the unification key
/// for the `eq_relations`; they carry a `TypeVariableValue` along
/// with them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct TyVidEqKey<'tcx> {
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
    fn index(&self) -> u32 { self.vid.index }
    fn from_index(i: u32) -> Self { TyVidEqKey::from(ty::TyVid { index: i }) }
    fn tag() -> &'static str { "TyVidEqKey" }
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

            // If both sides are unknown, we need to pick the most restrictive universe.
            (&TypeVariableValue::Unknown { universe: universe1 },
             &TypeVariableValue::Unknown { universe: universe2 }) => {
                let universe = cmp::min(universe1, universe2);
                Ok(TypeVariableValue::Unknown { universe })
            }
        }
    }
}

/// Raw `TyVid` are used as the unification key for `sub_relations`;
/// they carry no values.
impl ut::UnifyKey for ty::TyVid {
    type Value = ();
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::TyVid { ty::TyVid { index: i } }
    fn tag() -> &'static str { "TyVid" }
}

