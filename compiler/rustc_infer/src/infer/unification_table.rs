use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::undo_log::Rollback;
use rustc_data_structures::unify as ut;
use rustc_middle::ty;
use std::ops::Range;
use std::{fmt, marker::PhantomData};

use super::undo_log;
use super::InferCtxtUndoLogs;

pub trait InferVar: Into<u32> + From<u32> + fmt::Debug + Copy + PartialEq {
    type Value<'tcx>: Copy + fmt::Debug;
}

impl InferVar for ty::IntVid {
    type Value<'tcx> = ty::IntVarValue;
}

impl InferVar for ty::FloatVid {
    type Value<'tcx> = ty::FloatTy;
}

impl InferVar for ty::EffectVid {
    type Value<'tcx> = ty::Const<'tcx>;
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
pub(crate) enum UnifyValue<V> {
    Empty,
    Known(V),
}

impl<'tcx, K: InferVar> ut::UnifyKey for Key<'tcx, K> {
    type Value = UnifyValue<K::Value<'tcx>>;
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

impl<V: Copy + fmt::Debug> ut::UnifyValue for UnifyValue<V> {
    type Error = ut::NoError;

    fn unify_values(a: &UnifyValue<V>, b: &UnifyValue<V>) -> Result<Self, ut::NoError> {
        match (*a, *b) {
            (UnifyValue::Empty, UnifyValue::Empty) => Ok(UnifyValue::Empty),
            (UnifyValue::Known(value), UnifyValue::Empty)
            | (UnifyValue::Empty, UnifyValue::Known(value)) => Ok(UnifyValue::Known(value)),
            (UnifyValue::Known(a), UnifyValue::Known(b)) => {
                bug!("tried to instantiate an already known value {a:?} {b:?}")
            }
        }
    }
}

#[derive(Clone)]
pub struct NonRecursiveUnificationStorage<'tcx, K: InferVar> {
    storage: ut::UnificationTableStorage<Key<'tcx, K>>,
}

pub(crate) type UndoLogDelegate<'tcx, K> = sv::UndoLog<ut::Delegate<Key<'tcx, K>>>;

pub struct NonRecursiveUnificationTable<'a, 'tcx, K: InferVar> {
    table: ut::UnificationTable<
        ut::InPlace<
            Key<'tcx, K>,
            &'a mut ut::UnificationStorage<Key<'tcx, K>>,
            &'a mut InferCtxtUndoLogs<'tcx>,
        >,
    >,
}

impl<'tcx, K: InferVar> NonRecursiveUnificationStorage<'tcx, K> {
    pub fn new() -> NonRecursiveUnificationStorage<'tcx, K> {
        NonRecursiveUnificationStorage { storage: Default::default() }
    }

    #[inline]
    pub(super) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> NonRecursiveUnificationTable<'a, 'tcx, K>
    where
        undo_log::UndoLog<'tcx>: From<UndoLogDelegate<'tcx, K>>,
    {
        NonRecursiveUnificationTable { table: self.storage.with_log(undo_log) }
    }

    pub fn reverse(&mut self, undo: UndoLogDelegate<'tcx, K>) {
        self.storage.reverse(undo)
    }
}

impl<'tcx, K: InferVar> NonRecursiveUnificationTable<'_, 'tcx, K>
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

    pub fn new_key(&mut self) -> K {
        self.table.new_key(UnifyValue::Empty).value
    }

    #[inline(always)]
    pub fn inlined_probe_value(&mut self, id: K) -> Option<K::Value<'tcx>> {
        match self.table.probe_value(Key::new(id)) {
            UnifyValue::Empty => None,
            UnifyValue::Known(value) => Some(value),
        }
    }

    pub fn probe_value(&mut self, id: K) -> Option<K::Value<'tcx>> {
        match self.table.probe_value(Key::new(id)) {
            UnifyValue::Empty => None,
            UnifyValue::Known(value) => Some(value),
        }
    }

    pub fn unify(&mut self, a_id: K, b_id: K) {
        self.table.union(Key::new(a_id), Key::new(b_id))
    }

    pub fn instantiate(&mut self, id: K, value: K::Value<'tcx>) {
        self.table.union_value(Key::new(id), UnifyValue::Known(value))
    }
}
