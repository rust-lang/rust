// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::RelationDir::*;
use self::TypeVariableValue::*;
use self::UndoEntry::*;
use hir::def_id::{DefId};
use syntax::util::small_vector::SmallVector;
use syntax::ast;
use syntax_pos::Span;
use ty::{self, Ty};

use std::cmp::min;
use std::marker::PhantomData;
use std::mem;
use std::u32;
use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;

pub struct TypeVariableTable<'tcx> {
    values: sv::SnapshotVec<Delegate<'tcx>>,
    eq_relations: ut::UnificationTable<ty::TyVid>,
}

/// Reasons to create a type inference variable
pub enum TypeVariableOrigin {
    MiscVariable(Span),
    NormalizeProjectionType(Span),
    TypeInference(Span),
    TypeParameterDefinition(Span, ast::Name),
    TransformedUpvar(Span),
    SubstitutionPlaceholder(Span),
    AutoDeref(Span),
    AdjustmentType(Span),
    DivergingStmt(Span),
    DivergingBlockExpr(Span),
    LatticeVariable(Span),
}

struct TypeVariableData<'tcx> {
    value: TypeVariableValue<'tcx>,
    origin: TypeVariableOrigin,
    diverging: bool
}

enum TypeVariableValue<'tcx> {
    Known(Ty<'tcx>),
    Bounded {
        relations: Vec<Relation>,
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
}

enum UndoEntry<'tcx> {
    // The type of the var was specified.
    SpecifyVar(ty::TyVid, Vec<Relation>, Option<Default<'tcx>>),
    Relate(ty::TyVid, ty::TyVid),
    RelateRange(ty::TyVid, usize),
}

struct Delegate<'tcx>(PhantomData<&'tcx ()>);

type Relation = (RelationDir, ty::TyVid);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum RelationDir {
    SubtypeOf, SupertypeOf, EqTo, BiTo
}

impl RelationDir {
    fn opposite(self) -> RelationDir {
        match self {
            SubtypeOf => SupertypeOf,
            SupertypeOf => SubtypeOf,
            EqTo => EqTo,
            BiTo => BiTo,
        }
    }
}

impl<'tcx> TypeVariableTable<'tcx> {
    pub fn new() -> TypeVariableTable<'tcx> {
        TypeVariableTable {
            values: sv::SnapshotVec::new(),
            eq_relations: ut::UnificationTable::new(),
        }
    }

    fn relations<'a>(&'a mut self, a: ty::TyVid) -> &'a mut Vec<Relation> {
        relations(self.values.get_mut(a.index as usize))
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

    /// Records that `a <: b`, `a :> b`, or `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn relate_vars(&mut self, a: ty::TyVid, dir: RelationDir, b: ty::TyVid) {
        let a = self.root_var(a);
        let b = self.root_var(b);
        if a != b {
            if dir == EqTo {
                // a and b must be equal which we mark in the unification table
                let root = self.eq_relations.union(a, b);
                // In addition to being equal, all relations from the variable which is no longer
                // the root must be added to the root so they are not forgotten as the other
                // variable should no longer be referenced (other than to get the root)
                let other = if a == root { b } else { a };
                let count = {
                    let (relations, root_relations) = if other.index < root.index {
                        let (pre, post) = self.values.split_at_mut(root.index as usize);
                        (relations(&mut pre[other.index as usize]), relations(&mut post[0]))
                    } else {
                        let (pre, post) = self.values.split_at_mut(other.index as usize);
                        (relations(&mut post[0]), relations(&mut pre[root.index as usize]))
                    };
                    root_relations.extend_from_slice(relations);
                    relations.len()
                };
                self.values.record(RelateRange(root, count));
            } else {
                self.relations(a).push((dir, b));
                self.relations(b).push((dir.opposite(), a));
                self.values.record(Relate(a, b));
            }
        }
    }

    /// Instantiates `vid` with the type `ty` and then pushes an entry onto `stack` for each of the
    /// relations of `vid` to other variables. The relations will have the form `(ty, dir, vid1)`
    /// where `vid1` is some other variable id.
    ///
    /// Precondition: `vid` must be a root in the unification table
    pub fn instantiate_and_push(
        &mut self,
        vid: ty::TyVid,
        ty: Ty<'tcx>,
        stack: &mut SmallVector<(Ty<'tcx>, RelationDir, ty::TyVid)>)
    {
        debug_assert!(self.root_var(vid) == vid);
        let old_value = {
            let value_ptr = &mut self.values.get_mut(vid.index as usize).value;
            mem::replace(value_ptr, Known(ty))
        };

        let (relations, default) = match old_value {
            Bounded { relations, default } => (relations, default),
            Known(_) => bug!("Asked to instantiate variable that is \
                              already instantiated")
        };

        for &(dir, vid) in &relations {
            stack.push((ty, dir, vid));
        }

        self.values.record(SpecifyVar(vid, relations, default));
    }

    pub fn new_var(&mut self,
                   diverging: bool,
                   origin: TypeVariableOrigin,
                   default: Option<Default<'tcx>>,) -> ty::TyVid {
        self.eq_relations.new_key(());
        let index = self.values.push(TypeVariableData {
            value: Bounded { relations: vec![], default: default },
            origin: origin,
            diverging: diverging
        });
        let v = ty::TyVid { index: index as u32 };
        debug!("new_var() -> {:?}", v);
        v
    }

    pub fn num_vars(&self) -> usize {
        self.values.len()
    }

    pub fn root_var(&mut self, vid: ty::TyVid) -> ty::TyVid {
        self.eq_relations.find(vid)
    }

    pub fn probe(&mut self, vid: ty::TyVid) -> Option<Ty<'tcx>> {
        let vid = self.root_var(vid);
        self.probe_root(vid)
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

        self.values.rollback_to(s.snapshot);
        self.eq_relations.rollback_to(s.eq_snapshot);
    }

    pub fn commit(&mut self, s: Snapshot) {
        self.values.commit(s.snapshot);
        self.eq_relations.commit(s.eq_snapshot);
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

                sv::UndoLog::Other(SpecifyVar(vid, ..)) => {
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
    type Undo = UndoEntry<'tcx>;

    fn reverse(values: &mut Vec<TypeVariableData<'tcx>>, action: UndoEntry<'tcx>) {
        match action {
            SpecifyVar(vid, relations, default) => {
                values[vid.index as usize].value = Bounded {
                    relations: relations,
                    default: default
                };
            }

            Relate(a, b) => {
                relations(&mut (*values)[a.index as usize]).pop();
                relations(&mut (*values)[b.index as usize]).pop();
            }

            RelateRange(i, n) => {
                let relations = relations(&mut (*values)[i.index as usize]);
                for _ in 0..n {
                    relations.pop();
                }
            }
        }
    }
}

fn relations<'a>(v: &'a mut TypeVariableData) -> &'a mut Vec<Relation> {
    match v.value {
        Known(_) => bug!("var_sub_var: variable is known"),
        Bounded { ref mut relations, .. } => relations
    }
}
