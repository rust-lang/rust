use std::{marker::PhantomData, ops::Range};

use rustc_index::vec::Idx;

use crate::modified_set as ms;
use crate::snapshot_vec as sv;
use crate::unify as ut;
use crate::unify_log as ul;

use ena::undo_log::{Rollback, Snapshots, UndoLogs};

enum UndoLog<K: ut::UnifyKey, I> {
    Relation(sv::UndoLog<ut::Delegate<K>>),
    UnifyLog(ul::Undo<I>),
    ModifiedSet(ms::Undo),
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

impl<K: ut::UnifyKey, I> From<ms::Undo> for UndoLog<K, I> {
    fn from(l: ms::Undo) -> Self {
        UndoLog::ModifiedSet(l)
    }
}

struct Logs<K: ut::UnifyKey, I> {
    logs: Vec<UndoLog<K, I>>,
    num_open_snapshots: usize,
}

impl<K: ut::UnifyKey, I> Default for Logs<K, I> {
    fn default() -> Self {
        Self { logs: Default::default(), num_open_snapshots: Default::default() }
    }
}

impl<T, K: ut::UnifyKey, I> UndoLogs<T> for Logs<K, I>
where
    UndoLog<K, I>: From<T>,
{
    fn num_open_snapshots(&self) -> usize {
        self.num_open_snapshots
    }
    fn push(&mut self, undo: T) {
        if self.in_snapshot() {
            self.logs.push(undo.into())
        }
    }
    fn extend<J>(&mut self, undos: J)
    where
        Self: Sized,
        J: IntoIterator<Item = T>,
    {
        if self.in_snapshot() {
            self.logs.extend(undos.into_iter().map(UndoLog::from))
        }
    }
}

struct RollbackView<'a, K: ut::UnifyKey, I: Idx> {
    relations: &'a mut ut::UnificationStorage<K>,
    unify_log: &'a mut ul::UnifyLog<I>,
    modified_set: &'a mut ms::ModifiedSet<I>,
}

impl<K: ut::UnifyKey, I: Idx> Rollback<UndoLog<K, I>> for RollbackView<'_, K, I> {
    fn reverse(&mut self, undo: UndoLog<K, I>) {
        match undo {
            UndoLog::Relation(undo) => self.relations.reverse(undo),
            UndoLog::UnifyLog(undo) => self.unify_log.reverse(undo),
            UndoLog::ModifiedSet(undo) => self.modified_set.reverse(undo),
        }
    }
}

impl<K: ut::UnifyKey, I: Idx> Snapshots<UndoLog<K, I>> for Logs<K, I> {
    type Snapshot = Snapshot<K, I>;
    fn actions_since_snapshot(&self, snapshot: &Self::Snapshot) -> &[UndoLog<K, I>] {
        &self.logs[snapshot.undo_len..]
    }

    fn start_snapshot(&mut self) -> Self::Snapshot {
        unreachable!()
    }

    fn rollback_to(&mut self, values: &mut impl Rollback<UndoLog<K, I>>, snapshot: Self::Snapshot) {
        debug!("rollback_to({})", snapshot.undo_len);
        self.assert_open_snapshot(&snapshot);

        while self.logs.len() > snapshot.undo_len {
            values.reverse(self.logs.pop().unwrap());
        }

        if self.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.undo_len == 0);
            self.logs.clear();
        }

        self.num_open_snapshots -= 1;
    }

    fn commit(&mut self, snapshot: Self::Snapshot) {
        debug!("commit({})", snapshot.undo_len);

        if self.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.undo_len == 0);
            self.logs.clear();
        }

        self.num_open_snapshots -= 1;
    }
}

impl<K: ut::UnifyKey, I: Idx> Logs<K, I> {
    fn assert_open_snapshot(&self, snapshot: &Snapshot<K, I>) {
        // Failures here may indicate a failure to follow a stack discipline.
        assert!(self.logs.len() >= snapshot.undo_len);
        assert!(self.num_open_snapshots > 0);
    }
}

pub struct LoggedUnificationTable<K: ut::UnifyKey, I: Idx = K> {
    relations: ut::UnificationStorage<K>,
    unify_log: ul::UnifyLog<I>,
    modified_set: ms::ModifiedSet<I>,
    undo_log: Logs<K, I>,
}

impl<K, I> LoggedUnificationTable<K, I>
where
    K: ut::UnifyKey + From<I>,
    I: Idx + From<K>,
{
    pub fn new() -> Self {
        Self {
            relations: Default::default(),
            unify_log: ul::UnifyLog::new(),
            modified_set: ms::ModifiedSet::new(),
            undo_log: Logs::default(),
        }
    }

    fn relations(
        &mut self,
    ) -> ut::UnificationTable<ut::InPlace<K, &mut ut::UnificationStorage<K>, &mut Logs<K, I>>> {
        ut::UnificationTable::with_log(&mut self.relations, &mut self.undo_log)
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
        if self.unify_log.needs_log(vid) {
            warn!("ModifiedSet {:?} => {:?}", vid, ty);
            self.modified_set.set(&mut self.undo_log, vid);
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
        if self.unify_log.needs_log(vid) {
            self.modified_set.set(&mut self.undo_log, vid);
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
            self.unify_log.unify(&mut self.undo_log, a.into(), b.into());
        } else {
            self.unify_log.unify(&mut self.undo_log, b.into(), a.into());
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

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn vars_since_snapshot(&mut self, s: &Snapshot<K, I>) -> Range<K> {
        K::from(I::new(s.value_count))..K::from(I::new(self.relations().len()))
    }

    pub fn snapshot(&mut self) -> Snapshot<K, I> {
        self.undo_log.num_open_snapshots += 1;
        Snapshot {
            undo_len: self.undo_log.logs.len(),
            value_count: self.relations().len(),
            _marker: PhantomData,
        }
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot<K, I>) {
        let Self { relations, unify_log, modified_set, .. } = self;

        self.undo_log
            .rollback_to(&mut RollbackView { relations, unify_log, modified_set }, snapshot);

        if self.undo_log.num_open_snapshots == 0 {
            self.modified_set.clear();
        }
    }

    pub fn commit(&mut self, snapshot: Snapshot<K, I>) {
        self.undo_log.commit(snapshot);

        if self.undo_log.num_open_snapshots == 0 {
            self.modified_set.clear();
        }
    }

    pub fn register(&mut self) -> ms::Offset<I> {
        self.modified_set.register()
    }

    pub fn deregister(&mut self, offset: ms::Offset<I>) {
        self.modified_set.deregister(offset);
    }

    pub fn watch_variable(&mut self, index: I) {
        debug_assert!(index == self.relations().find(index).into());
        self.unify_log.watch_variable(index)
    }

    pub fn unwatch_variable(&mut self, index: I) {
        self.unify_log.unwatch_variable(index)
    }

    pub fn drain_modified_set(&mut self, offset: &ms::Offset<I>, mut f: impl FnMut(I) -> bool) {
        let unify_log = &self.unify_log;
        self.modified_set.drain(&mut self.undo_log, offset, |vid| {
            for &unified_vid in unify_log.get(vid) {
                f(unified_vid);
            }

            f(vid)
        })
    }
}

pub struct Snapshot<K: ut::UnifyKey, I: Idx = K> {
    undo_len: usize,
    value_count: usize,
    _marker: PhantomData<(K, I)>,
}
