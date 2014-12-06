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

use middle::ty::{mod, Ty};
use std::mem;
use util::snapshot_vec as sv;

pub struct TypeVariableTable<'tcx> {
    values: sv::SnapshotVec<TypeVariableData<'tcx>,UndoEntry,Delegate>,
}

struct TypeVariableData<'tcx> {
    value: TypeVariableValue<'tcx>,
    diverging: bool
}

enum TypeVariableValue<'tcx> {
    Known(Ty<'tcx>),
    Bounded(Vec<Relation>),
}

pub struct Snapshot {
    snapshot: sv::Snapshot
}

enum UndoEntry {
    // The type of the var was specified.
    SpecifyVar(ty::TyVid, Vec<Relation>),
    Relate(ty::TyVid, ty::TyVid),
}

struct Delegate;

type Relation = (RelationDir, ty::TyVid);

#[deriving(PartialEq,Show)]
pub enum RelationDir {
    SubtypeOf, SupertypeOf, EqTo
}

impl Copy for RelationDir {}

impl RelationDir {
    fn opposite(self) -> RelationDir {
        match self {
            SubtypeOf => SupertypeOf,
            SupertypeOf => SubtypeOf,
            EqTo => EqTo
        }
    }
}

impl<'tcx> TypeVariableTable<'tcx> {
    pub fn new() -> TypeVariableTable<'tcx> {
        TypeVariableTable { values: sv::SnapshotVec::new(Delegate) }
    }

    fn relations<'a>(&'a mut self, a: ty::TyVid) -> &'a mut Vec<Relation> {
        relations(self.values.get_mut(a.index))
    }

    pub fn var_diverges<'a>(&'a self, vid: ty::TyVid) -> bool {
        self.values.get(vid.index).diverging
    }

    /// Records that `a <: b`, `a :> b`, or `a == b`, depending on `dir`.
    ///
    /// Precondition: neither `a` nor `b` are known.
    pub fn relate_vars(&mut self, a: ty::TyVid, dir: RelationDir, b: ty::TyVid) {

        if a != b {
            self.relations(a).push((dir, b));
            self.relations(b).push((dir.opposite(), a));
            self.values.record(Relate(a, b));
        }
    }

    /// Instantiates `vid` with the type `ty` and then pushes an entry onto `stack` for each of the
    /// relations of `vid` to other variables. The relations will have the form `(ty, dir, vid1)`
    /// where `vid1` is some other variable id.
    pub fn instantiate_and_push(
        &mut self,
        vid: ty::TyVid,
        ty: Ty<'tcx>,
        stack: &mut Vec<(Ty<'tcx>, RelationDir, ty::TyVid)>)
    {
        let old_value = {
            let value_ptr = &mut self.values.get_mut(vid.index).value;
            mem::replace(value_ptr, Known(ty))
        };

        let relations = match old_value {
            Bounded(b) => b,
            Known(_) => panic!("Asked to instantiate variable that is \
                               already instantiated")
        };

        for &(dir, vid) in relations.iter() {
            stack.push((ty, dir, vid));
        }

        self.values.record(SpecifyVar(vid, relations));
    }

    pub fn new_var(&mut self, diverging: bool) -> ty::TyVid {
        let index = self.values.push(TypeVariableData {
            value: Bounded(vec![]),
            diverging: diverging
        });
        ty::TyVid { index: index }
    }

    pub fn probe(&self, vid: ty::TyVid) -> Option<Ty<'tcx>> {
        match self.values.get(vid.index).value {
            Bounded(..) => None,
            Known(t) => Some(t)
        }
    }

    pub fn replace_if_possible(&self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.sty {
            ty::ty_infer(ty::TyVar(v)) => {
                match self.probe(v) {
                    None => t,
                    Some(u) => u
                }
            }
            _ => t,
        }
    }

    pub fn snapshot(&mut self) -> Snapshot {
        Snapshot { snapshot: self.values.start_snapshot() }
    }

    pub fn rollback_to(&mut self, s: Snapshot) {
        self.values.rollback_to(s.snapshot);
    }

    pub fn commit(&mut self, s: Snapshot) {
        self.values.commit(s.snapshot);
    }
}

impl<'tcx> sv::SnapshotVecDelegate<TypeVariableData<'tcx>,UndoEntry> for Delegate {
    fn reverse(&mut self,
               values: &mut Vec<TypeVariableData>,
               action: UndoEntry) {
        match action {
            SpecifyVar(vid, relations) => {
                values[vid.index].value = Bounded(relations);
            }

            Relate(a, b) => {
                relations(&mut (*values)[a.index]).pop();
                relations(&mut (*values)[b.index]).pop();
            }
        }
    }
}

fn relations<'a>(v: &'a mut TypeVariableData) -> &'a mut Vec<Relation> {
    match v.value {
        Known(_) => panic!("var_sub_var: variable is known"),
        Bounded(ref mut relations) => relations
    }
}

