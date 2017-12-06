// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::TypeVariableValue::*;
use hir::def_id::{DefId};
use syntax::ast;
use syntax_pos::Span;
use ty::{self, Ty};

use std::cmp::min;
use std::marker::PhantomData;
use std::mem;
use std::u32;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;

pub struct TypeVariableTable<'tcx> {
    values: sv::SnapshotVec<Delegate<'tcx>>,

    /// Two variables are unified in `eq_relations` when we have a
    /// constraint `?X == ?Y`.
    eq_relations: ut::UnificationTable<ty::TyVid>,

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
    sub_relations: ut::UnificationTable<ty::TyVid>,
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

struct TypeVariableData<'tcx> {
    value: TypeVariableValue<'tcx>,
    origin: TypeVariableOrigin,
    diverging: bool
}

enum TypeVariableValue<'tcx> {
    Known(Ty<'tcx>),
    Bounded {
        default: Option<Default<'tcx>>
    }
}

// We will use this to store the required information to recapitulate what happened when
// an error occurs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Default<'tcx> {
    pub ty: Ty<'tcx>,
    /// The span where the default was incurred
    pub origin_span: Span,
    /// The definition that the default originates from
    pub def_id: DefId
}

pub struct Snapshot {
    snapshot: sv::Snapshot,
    eq_snapshot: ut::Snapshot<ty::TyVid>,
    sub_snapshot: ut::Snapshot<ty::TyVid>,
}

struct Instantiate<'tcx> {
    vid: ty::TyVid,
    default: Option<Default<'tcx>>,
}

struct Delegate<'tcx>(PhantomData<&'tcx ()>);

impl<'tcx> TypeVariableTable<'tcx> {
    pub fn new() -> TypeVariableTable<'tcx> {
        TypeVariableTable {
            values: sv::SnapshotVec::new(),
            eq_relations: ut::UnificationTable::new(),
            sub_relations: ut::UnificationTable::new(),
        }
    }

    pub fn default(&self, vid: ty::TyVid) -> Option<Default<'tcx>> {
        match &self.values.get(vid.index as usize).value {
            &Known(_) => None,
            &Bounded { ref default, .. } => default.clone()
        }
    }

    pub fn var_diverges<'a>(&'a self, vid: ty::TyVid) -> bool {
        self.values.get(vid.index as usize).diverging
    }

    pub fn var_origin(&self, vid: ty::TyVid) -> &TypeVariableOrigin {
        &self.values.get(vid.index as usize).origin
    }

    /// Records that `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn equate(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_none());
        debug_assert!(self.probe(b).is_none());
        self.eq_relations.union(a, b);
        self.sub_relations.union(a, b);
    }

    /// Records that `a <: b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn sub(&mut self, a: ty::TyVid, b: ty::TyVid) {
        debug_assert!(self.probe(a).is_none());
        debug_assert!(self.probe(b).is_none());
        self.sub_relations.union(a, b);
    }

    /// Instantiates `vid` with the type `ty`.
    ///
    /// Precondition: `vid` must not have been previously instantiated.
    pub fn instantiate(&mut self, vid: ty::TyVid, ty: Ty<'tcx>) {
        let vid = self.root_var(vid);
        debug_assert!(self.probe_root(vid).is_none());

        let old_value = {
            let vid_data = &mut self.values[vid.index as usize];
            mem::replace(&mut vid_data.value, TypeVariableValue::Known(ty))
        };

        match old_value {
            TypeVariableValue::Bounded { default } => {
                self.values.record(Instantiate { vid: vid, default: default });
            }
            TypeVariableValue::Known(old_ty) => {
                bug!("instantiating type variable `{:?}` twice: new-value = {:?}, old-value={:?}",
                     vid, ty, old_ty)
            }
        }
    }

    pub fn new_var(&mut self,
                   diverging: bool,
                   origin: TypeVariableOrigin,
                   default: Option<Default<'tcx>>,) -> ty::TyVid {
        debug!("new_var(diverging={:?}, origin={:?})", diverging, origin);
        self.eq_relations.new_key(());
        self.sub_relations.new_key(());
        let index = self.values.push(TypeVariableData {
            value: Bounded { default: default },
            origin,
            diverging,
        });
        let v = ty::TyVid { index: index as u32 };
        debug!("new_var: diverging={:?} index={:?}", diverging, v);
        v
    }

    pub fn num_vars(&self) -> usize {
        self.values.len()
    }

    /// Returns the "root" variable of `vid` in the `eq_relations`
    /// equivalence table. All type variables that have been equated
    /// will yield the same root variable (per the union-find
    /// algorithm), so `root_var(a) == root_var(b)` implies that `a ==
    /// b` (transitively).
    pub fn root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.eq_relations.find(vid)
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

    pub fn probe(&mut self, vid: ty::TyVid) -> Option<Ty<'tcx>> {
        let vid = self.root_var(vid);
        self.probe_root(vid)
    }

    pub fn origin(&self, vid: ty::TyVid) -> TypeVariableOrigin {
        self.values.get(vid.index as usize).origin.clone()
    }

    /// Retrieves the type of `vid` given that it is currently a root in the unification table
    pub fn probe_root(&mut self, vid: ty::TyVid) -> Option<Ty<'tcx>> {
        debug_assert!(self.root_var(vid) == vid);
        match self.values.get(vid.index as usize).value {
            Bounded { .. } => None,
            Known(t) => Some(t)
        }
    }

    pub fn replace_if_possible(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::TyInfer(ty::TyVar(v)) => {
                match self.probe(v) {
                    None => t,
                    Some(u) => u
                }
            }
            _ => t,
        }
    }

    pub fn snapshot(&mut self) -> Snapshot {
        Snapshot {
            snapshot: self.values.start_snapshot(),
            eq_snapshot: self.eq_relations.snapshot(),
            sub_snapshot: self.sub_relations.snapshot(),
        }
    }

    pub fn rollback_to(&mut self, s: Snapshot) {
        debug!("rollback_to{:?}", {
            for action in self.values.actions_since_snapshot(&s.snapshot) {
                match *action {
                    sv::UndoLog::NewElem(index) => {
                        debug!("inference variable _#{}t popped", index)
                    }
                    _ => { }
                }
            }
        });

        let Snapshot { snapshot, eq_snapshot, sub_snapshot } = s;
        self.values.rollback_to(snapshot);
        self.eq_relations.rollback_to(eq_snapshot);
        self.sub_relations.rollback_to(sub_snapshot);
    }

    pub fn commit(&mut self, s: Snapshot) {
        let Snapshot { snapshot, eq_snapshot, sub_snapshot } = s;
        self.values.commit(snapshot);
        self.eq_relations.commit(eq_snapshot);
        self.sub_relations.commit(sub_snapshot);
    }

    /// Returns a map `{V1 -> V2}`, where the keys `{V1}` are
    /// ty-variables created during the snapshot, and the values
    /// `{V2}` are the root variables that they were unified with,
    /// along with their origin.
    pub fn types_created_since_snapshot(&mut self, s: &Snapshot) -> TypeVariableMap {
        let actions_since_snapshot = self.values.actions_since_snapshot(&s.snapshot);

        actions_since_snapshot
            .iter()
            .filter_map(|action| match action {
                &sv::UndoLog::NewElem(index) => Some(ty::TyVid { index: index as u32 }),
                _ => None,
            })
            .map(|vid| {
                let origin = self.values.get(vid.index as usize).origin.clone();
                (vid, origin)
            })
            .collect()
    }

    pub fn types_escaping_snapshot(&mut self, s: &Snapshot) -> Vec<Ty<'tcx>> {
        /*!
         * Find the set of type variables that existed *before* `s`
         * but which have only been unified since `s` started, and
         * return the types with which they were unified. So if we had
         * a type variable `V0`, then we started the snapshot, then we
         * created a type variable `V1`, unifed `V0` with `T0`, and
         * unified `V1` with `T1`, this function would return `{T0}`.
         */

        let mut new_elem_threshold = u32::MAX;
        let mut escaping_types = Vec::new();
        let actions_since_snapshot = self.values.actions_since_snapshot(&s.snapshot);
        debug!("actions_since_snapshot.len() = {}", actions_since_snapshot.len());
        for action in actions_since_snapshot {
            match *action {
                sv::UndoLog::NewElem(index) => {
                    // if any new variables were created during the
                    // snapshot, remember the lower index (which will
                    // always be the first one we see). Note that this
                    // action must precede those variables being
                    // specified.
                    new_elem_threshold = min(new_elem_threshold, index as u32);
                    debug!("NewElem({}) new_elem_threshold={}", index, new_elem_threshold);
                }

                sv::UndoLog::Other(Instantiate { vid, .. }) => {
                    if vid.index < new_elem_threshold {
                        // quick check to see if this variable was
                        // created since the snapshot started or not.
                        let escaping_type = match self.values.get(vid.index as usize).value {
                            Bounded { .. } => bug!(),
                            Known(ty) => ty,
                        };
                        escaping_types.push(escaping_type);
                    }
                    debug!("SpecifyVar({:?}) new_elem_threshold={}", vid, new_elem_threshold);
                }

                _ => { }
            }
        }

        escaping_types
    }

    pub fn unsolved_variables(&mut self) -> Vec<ty::TyVid> {
        (0..self.values.len())
            .filter_map(|i| {
                let vid = ty::TyVid { index: i as u32 };
                if self.probe(vid).is_some() {
                    None
                } else {
                    Some(vid)
                }
            })
            .collect()
    }
}

impl<'tcx> sv::SnapshotVecDelegate for Delegate<'tcx> {
    type Value = TypeVariableData<'tcx>;
    type Undo = Instantiate<'tcx>;

    fn reverse(values: &mut Vec<TypeVariableData<'tcx>>, action: Instantiate<'tcx>) {
        let Instantiate { vid, default } = action;
        values[vid.index as usize].value = Bounded {
            default,
        };
    }
}
