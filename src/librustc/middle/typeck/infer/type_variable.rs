// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty;
use std::mem;
use util::snapshot_vec as sv;

pub struct TypeVariableTable {
    values: sv::SnapshotVec<TypeVariableData,UndoEntry,Delegate>,
}

struct TypeVariableData {
    value: TypeVariableValue
}

enum TypeVariableValue {
    Known(ty::t),
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

impl RelationDir {
    fn opposite(self) -> RelationDir {
        match self {
            SubtypeOf => SupertypeOf,
            SupertypeOf => SubtypeOf,
            EqTo => EqTo
        }
    }
}

impl TypeVariableTable {
    pub fn new() -> TypeVariableTable {
        TypeVariableTable { values: sv::SnapshotVec::new(Delegate) }
    }

    fn relations<'a>(&'a mut self, a: ty::TyVid) -> &'a mut Vec<Relation> {
        relations(self.values.get_mut(a.index))
    }

    pub fn relate_vars(&mut self, a: ty::TyVid, dir: RelationDir, b: ty::TyVid) {
        /*!
         * Records that `a <: b`, `a :> b`, or `a == b`, depending on `dir`.
         *
         * Precondition: neither `a` nor `b` are known.
         */

        if a != b {
            self.relations(a).push((dir, b));
            self.relations(b).push((dir.opposite(), a));
            self.values.record(Relate(a, b));
        }
    }

    pub fn instantiate_and_push(
        &mut self,
        vid: ty::TyVid,
        ty: ty::t,
        stack: &mut Vec<(ty::t, RelationDir, ty::TyVid)>)
    {
        /*!
         * Instantiates `vid` with the type `ty` and then pushes an
         * entry onto `stack` for each of the relations of `vid` to
         * other variables. The relations will have the form `(ty,
         * dir, vid1)` where `vid1` is some other variable id.
         */

        let old_value = {
            let value_ptr = &mut self.values.get_mut(vid.index).value;
            mem::replace(value_ptr, Known(ty))
        };

        let relations = match old_value {
            Bounded(b) => b,
            Known(_) => fail!("Asked to instantiate variable that is \
                               already instantiated")
        };

        for &(dir, vid) in relations.iter() {
            stack.push((ty, dir, vid));
        }

        self.values.record(SpecifyVar(vid, relations));
    }

    pub fn new_var(&mut self) -> ty::TyVid {
        let index =
            self.values.push(
                TypeVariableData { value: Bounded(Vec::new()) });
        ty::TyVid { index: index }
    }

    pub fn probe(&self, vid: ty::TyVid) -> Option<ty::t> {
        match self.values.get(vid.index).value {
            Bounded(..) => None,
            Known(t) => Some(t)
        }
    }

    pub fn replace_if_possible(&self, t: ty::t) -> ty::t {
        match ty::get(t).sty {
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

impl sv::SnapshotVecDelegate<TypeVariableData,UndoEntry> for Delegate {
    fn reverse(&mut self,
               values: &mut Vec<TypeVariableData>,
               action: UndoEntry) {
        match action {
            SpecifyVar(vid, relations) => {
                values.get_mut(vid.index).value = Bounded(relations);
            }

            Relate(a, b) => {
                relations(values.get_mut(a.index)).pop();
                relations(values.get_mut(b.index)).pop();
            }
        }
    }
}

fn relations<'a>(v: &'a mut TypeVariableData) -> &'a mut Vec<Relation> {
    match v.value {
        Known(_) => fail!("var_sub_var: variable is known"),
        Bounded(ref mut relations) => relations
    }
}

