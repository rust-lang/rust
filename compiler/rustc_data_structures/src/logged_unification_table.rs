use rustc_index::vec::Idx;

use crate::modified_set as ms;
use crate::snapshot_vec as sv;
use crate::unify as ut;
use crate::unify_log as ul;

use ena::undo_log::{Rollback, UndoLogs};

pub enum UndoLog<K: ut::UnifyKey, I = K> {
    Relation(sv::UndoLog<ut::Delegate<K>>),
    UnifyLog(ul::Undo<I>),
    ModifiedSet(ms::Undo<I>),
}

impl<K: ut::UnifyKey, I> From<sv::UndoLog<ut::Delegate<K>>> for UndoLog<K, I> {
    fn from(l: sv::UndoLog<ut::Delegate<K>>) -> Self {
        UndoLog::Relation(l)
    }
}

impl<K: ut::UnifyKey, I> From<ul::Undo<I>> for UndoLog<K, I> {
    fn from(l: ul::Undo<I>) -> Self {
        UndoLog::UnifyLog(l)
    }
}

impl<K: ut::UnifyKey, I> From<ms::Undo<I>> for UndoLog<K, I> {
    fn from(l: ms::Undo<I>) -> Self {
        UndoLog::ModifiedSet(l)
    }
}

impl<K: ut::UnifyKey, I: Idx> Rollback<UndoLog<K, I>> for LoggedUnificationTableStorage<K, I> {
    fn reverse(&mut self, undo: UndoLog<K, I>) {
        match undo {
            UndoLog::Relation(undo) => self.relations.reverse(undo),
            UndoLog::UnifyLog(undo) => self.unify_log.reverse(undo),
            UndoLog::ModifiedSet(undo) => self.modified_set.reverse(undo),
        }
    }
}

pub struct LoggedUnificationTableStorage<K: ut::UnifyKey, I: Idx = K> {
    relations: ut::UnificationTableStorage<K>,
    unify_log: ul::UnifyLog<I>,
    modified_set: ms::ModifiedSet<I>,
}

pub struct LoggedUnificationTable<'a, K: ut::UnifyKey, I: Idx, L> {
    storage: &'a mut LoggedUnificationTableStorage<K, I>,
    undo_log: L,
}

impl<K, I> LoggedUnificationTableStorage<K, I>
where
    K: ut::UnifyKey + From<I>,
    I: Idx + From<K>,
{
    pub fn new() -> Self {
        Self {
            relations: Default::default(),
            unify_log: ul::UnifyLog::new(),
            modified_set: ms::ModifiedSet::new(),
        }
    }

    pub fn with_log<L>(&mut self, undo_log: L) -> LoggedUnificationTable<'_, K, I, L> {
        LoggedUnificationTable { storage: self, undo_log }
    }
}

impl<K, I, L> LoggedUnificationTable<'_, K, I, L>
where
    K: ut::UnifyKey,
    I: Idx,
{
    pub fn len(&self) -> usize {
        self.storage.relations.len()
    }
}

impl<K, I, L> LoggedUnificationTable<'_, K, I, L>
where
    K: ut::UnifyKey + From<I>,
    I: Idx + From<K>,
    L: UndoLogs<ms::Undo<I>> + UndoLogs<ul::Undo<I>> + UndoLogs<sv::UndoLog<ut::Delegate<K>>>,
{
    fn relations(
        &mut self,
    ) -> ut::UnificationTable<ut::InPlace<K, &mut ut::UnificationStorage<K>, &mut L>> {
        ut::UnificationTable::with_log(&mut self.storage.relations, &mut self.undo_log)
    }

    pub fn unify(&mut self, a: I, b: I)
    where
        K::Value: ut::UnifyValue<Error = ut::NoError>,
    {
        self.unify_var_var(a, b).unwrap();
    }

    pub fn instantiate(&mut self, vid: I, ty: K::Value) -> K
    where
        K::Value: ut::UnifyValue<Error = ut::NoError>,
    {
        if self.storage.unify_log.needs_log(vid) {
            warn!("ModifiedSet {:?} => {:?}", vid, ty);
            self.storage.modified_set.set(&mut self.undo_log, vid);
        }
        let vid = vid.into();
        let mut relations = self.relations();
        debug_assert!(relations.find(vid) == vid);
        relations.union_value(vid, ty);

        vid
    }

    pub fn find(&mut self, vid: I) -> K {
        self.relations().find(vid)
    }

    pub fn unify_var_value(
        &mut self,
        vid: I,
        value: K::Value,
    ) -> Result<(), <K::Value as ut::UnifyValue>::Error> {
        let vid = self.find(vid).into();
        if self.storage.unify_log.needs_log(vid) {
            self.storage.modified_set.set(&mut self.undo_log, vid);
        }
        self.relations().unify_var_value(vid, value)
    }

    pub fn unify_var_var(&mut self, a: I, b: I) -> Result<(), <K::Value as ut::UnifyValue>::Error> {
        let mut relations = self.relations();
        let a = relations.find(a);
        let b = relations.find(b);
        if a == b {
            return Ok(());
        }

        relations.unify_var_var(a, b)?;

        if a == relations.find(a) {
            self.storage.unify_log.unify(&mut self.undo_log, a.into(), b.into());
        } else {
            self.storage.unify_log.unify(&mut self.undo_log, b.into(), a.into());
        }
        Ok(())
    }

    pub fn union_value(&mut self, vid: I, value: K::Value)
    where
        K::Value: ut::UnifyValue<Error = ut::NoError>,
    {
        let vid = self.find(vid).into();
        self.instantiate(vid, value);
    }

    pub fn probe_value(&mut self, vid: I) -> K::Value {
        self.relations().probe_value(vid)
    }

    #[inline(always)]
    pub fn inlined_probe_value(&mut self, vid: I) -> K::Value {
        self.relations().inlined_probe_value(vid)
    }

    pub fn new_key(&mut self, value: K::Value) -> K {
        self.relations().new_key(value)
    }

    pub fn clear_modified_set(&mut self) {
        self.storage.modified_set.clear();
    }

    pub fn register(&mut self) -> ms::Offset<I> {
        self.storage.modified_set.register()
    }

    pub fn deregister(&mut self, offset: ms::Offset<I>) {
        self.storage.modified_set.deregister(offset);
    }

    pub fn watch_variable(&mut self, index: I) {
        debug_assert!(index == self.relations().find(index).into());
        self.storage.unify_log.watch_variable(index)
    }

    pub fn unwatch_variable(&mut self, index: I) {
        self.storage.unify_log.unwatch_variable(index)
    }

    pub fn drain_modified_set(&mut self, offset: &ms::Offset<I>, mut f: impl FnMut(I) -> bool) {
        let unify_log = &self.storage.unify_log;
        self.storage.modified_set.drain(&mut self.undo_log, offset, |vid| {
            for &unified_vid in unify_log.get(vid) {
                f(unified_vid);
            }

            f(vid)
        })
    }
}
