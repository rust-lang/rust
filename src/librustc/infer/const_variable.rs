use crate::mir::interpret::ConstValue;
use syntax::symbol::InternedString;
use syntax_pos::Span;
use crate::ty::{self, InferConst};

use std::cmp;
use std::marker::PhantomData;
use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;

pub struct ConstVariableTable<'tcx> {
    values: sv::SnapshotVec<Delegate<'tcx>>,

    relations: ut::UnificationTable<ut::InPlace<ty::ConstVid<'tcx>>>,
}

/// Reasons to create a const inference variable
#[derive(Copy, Clone, Debug)]
pub enum ConstVariableOrigin {
    MiscVariable(Span),
    ConstInference(Span),
    ConstParameterDefinition(Span, InternedString),
    SubstitutionPlaceholder(Span),
}

struct ConstVariableData {
    origin: ConstVariableOrigin,
}

#[derive(Copy, Clone, Debug)]
pub enum ConstVariableValue<'tcx> {
    Known { value: &'tcx ty::LazyConst<'tcx> },
    Unknown { universe: ty::UniverseIndex },
}

impl<'tcx> ConstVariableValue<'tcx> {
    /// If this value is known, returns the const it is known to be.
    /// Otherwise, `None`.
    pub fn known(&self) -> Option<&'tcx ty::LazyConst<'tcx>> {
        match *self {
            ConstVariableValue::Unknown { .. } => None,
            ConstVariableValue::Known { value } => Some(value),
        }
    }

    pub fn is_unknown(&self) -> bool {
        match *self {
            ConstVariableValue::Unknown { .. } => true,
            ConstVariableValue::Known { .. } => false,
        }
    }
}

pub struct Snapshot<'tcx> {
    snapshot: sv::Snapshot,
    relation_snapshot: ut::Snapshot<ut::InPlace<ty::ConstVid<'tcx>>>,
}

struct Instantiate<'tcx> {
    _vid: ty::ConstVid<'tcx>,
}

struct Delegate<'tcx> {
    pub phantom: PhantomData<&'tcx ()>,
}

impl<'tcx> ConstVariableTable<'tcx> {
    pub fn new() -> ConstVariableTable<'tcx> {
        ConstVariableTable {
            values: sv::SnapshotVec::new(),
            relations: ut::UnificationTable::new(),
        }
    }

    /// Returns the origin that was given when `vid` was created.
    ///
    /// Note that this function does not return care whether
    /// `vid` has been unified with something else or not.
    pub fn var_origin(&self, vid: ty::ConstVid<'tcx>) -> &ConstVariableOrigin {
        &self.values[vid.index as usize].origin
    }

    pub fn unify_var_var(
        &mut self,
        a_id: ty::ConstVid<'tcx>,
        b_id: ty::ConstVid<'tcx>,
    ) -> Result<(), (&'tcx ty::LazyConst<'tcx>, &'tcx ty::LazyConst<'tcx>)> {
        self.relations.unify_var_var(a_id, b_id)
    }

    pub fn unify_var_value(
        &mut self,
        a_id: ty::ConstVid<'tcx>,
        b: ConstVariableValue<'tcx>,
    ) -> Result<(), (&'tcx ty::LazyConst<'tcx>, &'tcx ty::LazyConst<'tcx>)> {
        self.relations.unify_var_value(a_id, b)
    }

    /// Creates a new const variable.
    ///
    /// - `origin`: indicates *why* the const variable was created.
    ///   The code in this module doesn't care, but it can be useful
    ///   for improving error messages.
    pub fn new_var(
        &mut self,
        universe: ty::UniverseIndex,
        origin: ConstVariableOrigin,
    ) -> ty::ConstVid<'tcx> {
        let vid = self.relations.new_key(ConstVariableValue::Unknown{ universe });

        let index = self.values.push(ConstVariableData {
            origin,
        });
        assert_eq!(vid.index, index as u32);

        debug!("new_var(index={:?}, origin={:?}", vid, origin);

        vid
    }

    /// Retrieves the type to which `vid` has been instantiated, if
    /// any.
    pub fn probe(
        &mut self,
        vid: ty::ConstVid<'tcx>
    ) -> ConstVariableValue<'tcx> {
        self.relations.probe_value(vid)
    }

    /// If `t` is a type-inference variable, and it has been
    /// instantiated, then return the with which it was
    /// instantiated. Otherwise, returns `t`.
    pub fn replace_if_possible(
        &mut self,
        c: &'tcx ty::LazyConst<'tcx>
    ) -> &'tcx ty::LazyConst<'tcx> {
        if let ty::LazyConst::Evaluated(ty::Const {
            val: ConstValue::Infer(InferConst::Var(vid)),
            ..
        }) = c {
            match self.probe(*vid).known() {
                Some(c) => c,
                None => c,
            }
        } else {
            c
        }
    }

    /// Creates a snapshot of the type variable state.  This snapshot
    /// must later be committed (`commit()`) or rolled back
    /// (`rollback_to()`).  Nested snapshots are permitted, but must
    /// be processed in a stack-like fashion.
    pub fn snapshot(&mut self) -> Snapshot<'tcx> {
        Snapshot {
            snapshot: self.values.start_snapshot(),
            relation_snapshot: self.relations.snapshot(),
        }
    }

    /// Undoes all changes since the snapshot was created. Any
    /// snapshots created since that point must already have been
    /// committed or rolled back.
    pub fn rollback_to(&mut self, s: Snapshot<'tcx>) {
        debug!("rollback_to{:?}", {
            for action in self.values.actions_since_snapshot(&s.snapshot) {
                if let sv::UndoLog::NewElem(index) = *action {
                    debug!("inference variable _#{}t popped", index)
                }
            }
        });

        let Snapshot { snapshot, relation_snapshot } = s;
        self.values.rollback_to(snapshot);
        self.relations.rollback_to(relation_snapshot);
    }

    /// Commits all changes since the snapshot was created, making
    /// them permanent (unless this snapshot was created within
    /// another snapshot). Any snapshots created since that point
    /// must already have been committed or rolled back.
    pub fn commit(&mut self, s: Snapshot<'tcx>) {
        let Snapshot { snapshot, relation_snapshot } = s;
        self.values.commit(snapshot);
        self.relations.commit(relation_snapshot);
    }
}

impl<'tcx> ut::UnifyKey for ty::ConstVid<'tcx> {
    type Value = ConstVariableValue<'tcx>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> Self { ty::ConstVid { index: i, phantom: PhantomData } }
    fn tag() -> &'static str { "ConstVid" }
}

impl<'tcx> ut::UnifyValue for ConstVariableValue<'tcx> {
    type Error = (&'tcx ty::LazyConst<'tcx>, &'tcx ty::LazyConst<'tcx>);

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        match (value1, value2) {
            (
                &ConstVariableValue::Known { value: value1 },
                &ConstVariableValue::Known { value: value2 }
            ) => {
                match <&'tcx ty::LazyConst<'tcx>>::unify_values(&value1, &value2) {
                    Ok(value) => Ok(ConstVariableValue::Known { value }),
                    Err(err) => Err(err),
                }
            }

            // If one side is known, prefer that one.
            (&ConstVariableValue::Known { .. }, &ConstVariableValue::Unknown { .. }) => Ok(*value1),
            (&ConstVariableValue::Unknown { .. }, &ConstVariableValue::Known { .. }) => Ok(*value2),

            // If both sides are *unknown*, it hardly matters, does it?
            (&ConstVariableValue::Unknown { universe: universe1 },
             &ConstVariableValue::Unknown { universe: universe2 }) =>  {
                // If we unify two unbound variables, ?T and ?U, then whatever
                // value they wind up taking (which must be the same value) must
                // be nameable by both universes. Therefore, the resulting
                // universe is the minimum of the two universes, because that is
                // the one which contains the fewest names in scope.
                let universe = cmp::min(universe1, universe2);
                Ok(ConstVariableValue::Unknown { universe })
            }
        }
    }
}

impl<'tcx> ut::EqUnifyValue for &'tcx ty::LazyConst<'tcx> {}

impl<'tcx> sv::SnapshotVecDelegate for Delegate<'tcx> {
    type Value = ConstVariableData;
    type Undo = Instantiate<'tcx>;

    fn reverse(_values: &mut Vec<ConstVariableData>, _action: Instantiate<'tcx>) {
        // We don't actually have to *do* anything to reverse an
        // instantiation; the value for a variable is stored in the
        // `relations` and hence its rollback code will handle
        // it.
    }
}
