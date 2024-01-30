use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::undo_log::Rollback;
use rustc_data_structures::unify as ut;
use rustc_middle::infer::unify_key::ConstVariableOrigin;
use rustc_middle::ty::{self, Ty};
use std::cmp;
use std::ops::Range;
use std::{fmt, marker::PhantomData};

use super::undo_log;
use super::InferCtxtUndoLogs;

pub trait InferVar: Into<u32> + From<u32> + fmt::Debug + Copy + PartialEq {
    type Value<'tcx>: Copy + fmt::Debug;
    type VarData<'tcx>: Copy + fmt::Debug;
    fn merge_data<'tcx>(lhs: Self::VarData<'tcx>, rhs: Self::VarData<'tcx>) -> Self::VarData<'tcx>;
}

impl InferVar for ty::IntVid {
    type Value<'tcx> = ty::IntVarValue;
    type VarData<'tcx> = ();
    fn merge_data<'tcx>(_: Self::VarData<'tcx>, _: Self::VarData<'tcx>) -> Self::VarData<'tcx> {}
}

impl InferVar for ty::FloatVid {
    type Value<'tcx> = ty::FloatTy;
    type VarData<'tcx> = ();
    fn merge_data<'tcx>(_: Self::VarData<'tcx>, _: Self::VarData<'tcx>) -> Self::VarData<'tcx> {}
}

impl InferVar for ty::EffectVid {
    type Value<'tcx> = ty::Const<'tcx>;
    type VarData<'tcx> = ();
    fn merge_data<'tcx>(_: Self::VarData<'tcx>, _: Self::VarData<'tcx>) -> Self::VarData<'tcx> {}
}

impl InferVar for ty::ConstVid {
    type Value<'tcx> = ty::Const<'tcx>;
    type VarData<'tcx> = (ConstVariableOrigin, ty::UniverseIndex);
    fn merge_data<'tcx>(
        (origin, lhs): Self::VarData<'tcx>,
        (_origin, rhs): Self::VarData<'tcx>,
    ) -> Self::VarData<'tcx> {
        // If we unify two unbound variables, ?T and ?U, then whatever
        // value they wind up taking (which must be the same value) must
        // be nameable by both universes. Therefore, the resulting
        // universe is the minimum of the two universes, because that is
        // the one which contains the fewest names in scope.
        (origin, cmp::min(lhs, rhs))
    }
}

impl InferVar for ty::TyVid {
    type Value<'tcx> = Ty<'tcx>;
    type VarData<'tcx> = ty::UniverseIndex;
    fn merge_data<'tcx>(lhs: Self::VarData<'tcx>, rhs: Self::VarData<'tcx>) -> Self::VarData<'tcx> {
        // If we unify two unbound variables, ?T and ?U, then whatever
        // value they wind up taking (which must be the same value) must
        // be nameable by both universes. Therefore, the resulting
        // universe is the minimum of the two universes, because that is
        // the one which contains the fewest names in scope.
        cmp::min(lhs, rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Key<'tcx, K> {
    value: K,
    _marker: PhantomData<&'tcx K>,
}

impl<'tcx, K> Key<'tcx, K> {
    #[inline]
    fn new(value: K) -> Key<'tcx, K> {
        Key { value, _marker: PhantomData }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VariableValue<'tcx, K: InferVar> {
    Unknown(K::VarData<'tcx>),
    Known(K::Value<'tcx>),
}

impl<'tcx, K: InferVar> VariableValue<'tcx, K> {
    pub(crate) fn known(self) -> Option<K::Value<'tcx>> {
        match self {
            VariableValue::Unknown(_) => None,
            VariableValue::Known(value) => Some(value),
        }
    }

    pub(crate) fn is_unknown(self) -> bool {
        match self {
            VariableValue::Unknown(_) => true,
            VariableValue::Known(_) => false,
        }
    }

    pub(crate) fn is_known(self) -> bool {
        match self {
            VariableValue::Unknown(_) => false,
            VariableValue::Known(_) => true,
        }
    }
}

impl<'tcx, K: InferVar> ut::UnifyKey for Key<'tcx, K> {
    type Value = VariableValue<'tcx, K>;
    #[inline]
    fn index(&self) -> u32 {
        self.value.into()
    }
    #[inline]
    fn from_index(i: u32) -> Self {
        Key { value: i.into(), _marker: PhantomData }
    }
    fn tag() -> &'static str {
        std::any::type_name::<K>()
    }
}

impl<'tcx, K: InferVar> ut::UnifyValue for VariableValue<'tcx, K> {
    type Error = ut::NoError;

    fn unify_values(
        a: &VariableValue<'tcx, K>,
        b: &VariableValue<'tcx, K>,
    ) -> Result<Self, ut::NoError> {
        match (*a, *b) {
            (VariableValue::Unknown(a_data), VariableValue::Unknown(b_data)) => {
                Ok(VariableValue::Unknown(K::merge_data(a_data, b_data)))
            }
            (VariableValue::Known(value), VariableValue::Unknown(_))
            | (VariableValue::Unknown(_), VariableValue::Known(value)) => {
                Ok(VariableValue::Known(value))
            }
            (VariableValue::Known(a), VariableValue::Known(b)) => {
                bug!("tried to instantiate an already known value {a:?} {b:?}")
            }
        }
    }
}

#[derive(Clone)]
pub struct UnificationStorage<'tcx, K: InferVar> {
    storage: ut::UnificationTableStorage<Key<'tcx, K>>,
}

pub(crate) type UndoLogDelegate<'tcx, K> = sv::UndoLog<ut::Delegate<Key<'tcx, K>>>;

pub struct UnificationTable<'a, 'tcx, K: InferVar> {
    table: ut::UnificationTable<
        ut::InPlace<
            Key<'tcx, K>,
            &'a mut ut::UnificationStorage<Key<'tcx, K>>,
            &'a mut InferCtxtUndoLogs<'tcx>,
        >,
    >,
}

impl<'tcx, K: InferVar> UnificationStorage<'tcx, K> {
    pub fn new() -> UnificationStorage<'tcx, K> {
        UnificationStorage { storage: Default::default() }
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    #[inline]
    pub(super) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> UnificationTable<'a, 'tcx, K>
    where
        undo_log::UndoLog<'tcx>: From<UndoLogDelegate<'tcx, K>>,
    {
        UnificationTable { table: self.storage.with_log(undo_log) }
    }

    pub(super) fn try_probe_value(&self, id: K) -> Option<&VariableValue<'tcx, K>> {
        self.storage.try_probe_value(Key::new(id))
    }

    pub fn reverse(&mut self, undo: UndoLogDelegate<'tcx, K>) {
        self.storage.reverse(undo)
    }
}

impl<'tcx, K: InferVar> UnificationTable<'_, 'tcx, K>
where
    undo_log::UndoLog<'tcx>: From<UndoLogDelegate<'tcx, K>>,
{
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// HACK: Used by `fudge_inference_if_ok` to restore inference variables.
    pub fn vars_since_snapshot(&self, snapshot_var_len: usize) -> Range<K> {
        K::from(snapshot_var_len as u32)..K::from(self.len() as u32)
    }

    pub fn current_root(&mut self, id: K) -> K {
        self.table.find(Key::new(id)).value
    }

    pub fn new_key(&mut self, data: K::VarData<'tcx>) -> K {
        self.table.new_key(VariableValue::Unknown(data)).value
    }

    #[inline(always)]
    pub fn inlined_probe_value(&mut self, id: K) -> VariableValue<'tcx, K> {
        self.table.inlined_probe_value(Key::new(id))
    }

    pub fn probe_value(&mut self, id: K) -> VariableValue<'tcx, K> {
        self.table.probe_value(Key::new(id))
    }

    pub fn unify(&mut self, a_id: K, b_id: K) {
        self.table.union(Key::new(a_id), Key::new(b_id))
    }

    pub fn unified(&mut self, a_id: K, b_id: K) -> bool {
        self.table.unioned(Key::new(a_id), Key::new(b_id))
    }

    pub fn instantiate(&mut self, id: K, value: K::Value<'tcx>) {
        self.table.union_value(Key::new(id), VariableValue::Known(value))
    }
}
