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
use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;

pub struct TypeVariableTable<'tcx> {
    values: sv::SnapshotVec<Delegate<'tcx>>,
    eq_relations: ut::UnificationTable<ty::TyVid>,
}

/// Reasons to create a type inference variable
#[derive(Debug)]
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
    DivergingFn(Span),
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
    }

    /// Instantiates `vid` with the type `ty`.
    ///
    /// Precondition: `vid` must be a root in the unification table
    /// and has not previously been instantiated.
    pub fn instantiate(&mut self, vid: ty::TyVid, ty: Ty<'tcx>) {
        debug_assert!(self.root_var(vid) == vid);
        debug_assert!(self.probe(vid).is_none());

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
        let index = self.values.push(TypeVariableData {
            value: Bounded { default: default },
            origin: origin,
            diverging: diverging
        });
        let v = ty::TyVid { index: index as u32 };
        debug!("new_var: diverging={:?} index={:?}", diverging, v);
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
            default: default
        };
    }
}
