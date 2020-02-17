use std::ops::Range;

use rustc_index::vec::{Idx, IndexVec};

use crate::modified_set as ms;
use crate::unify as ut;
use crate::unify_log as ul;

pub struct LoggedUnificationTable<K: ut::UnifyKey, I: Idx = K> {
    relations: ut::UnificationTable<ut::InPlace<K>>,
    unify_log: ul::UnifyLog<I>,
    modified_set: ms::ModifiedSet<I>,
    reference_counts: IndexVec<I, u32>,
}

impl<K, I> LoggedUnificationTable<K, I>
where
    K: ut::UnifyKey + From<I>,
    I: Idx + From<K>,
{
    pub fn new() -> Self {
        Self {
            relations: ut::UnificationTable::new(),
            unify_log: ul::UnifyLog::new(),
            modified_set: ms::ModifiedSet::new(),
            reference_counts: IndexVec::new(),
        }
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
        if self.needs_log(vid) {
            warn!("ModifiedSet {:?} => {:?}", vid, ty);
            self.modified_set.set(vid);
        }
        let vid = vid.into();
        debug_assert!(self.relations.find(vid) == vid);
        self.relations.union_value(vid, ty);

        vid
    }

    pub fn find(&mut self, vid: I) -> K {
        self.relations.find(vid)
    }

    pub fn unify_var_value(
        &mut self,
        vid: I,
        value: K::Value,
    ) -> Result<(), <K::Value as ut::UnifyValue>::Error> {
        let vid = self.find(vid).into();
        if self.needs_log(vid) {
            self.modified_set.set(vid);
        }
        self.relations.unify_var_value(vid, value)
    }

    pub fn unify_var_var(&mut self, a: I, b: I) -> Result<(), <K::Value as ut::UnifyValue>::Error> {
        let a = self.relations.find(a);
        let b = self.relations.find(b);
        if a == b {
            return Ok(());
        }

        self.relations.unify_var_var(a, b)?;

        if self.needs_log(a.into()) || self.needs_log(b.into()) {
            warn!("Log: {:?} {:?} => {:?}", a, b, I::from(self.relations.find(a)));
            if a == self.relations.find(a) {
                self.unify_log.unify(a.into(), b.into());
            } else {
                self.unify_log.unify(b.into(), a.into());
            }
        }
        Ok(())
    }

    fn needs_log(&self, vid: I) -> bool {
        !self.unify_log.get(vid).is_empty()
            || self.reference_counts.get(vid).map_or(false, |c| *c != 0)
    }

    pub fn union_value(&mut self, vid: I, value: K::Value)
    where
        K::Value: ut::UnifyValue<Error = ut::NoError>,
    {
        let vid = self.find(vid).into();
        self.instantiate(vid, value);
    }

    pub fn probe_value(&mut self, vid: I) -> K::Value {
        self.relations.probe_value(vid)
    }

    #[inline(always)]
    pub fn inlined_probe_value(&mut self, vid: I) -> K::Value {
        self.relations.inlined_probe_value(vid)
    }

    pub fn new_key(&mut self, value: K::Value) -> K {
        self.relations.new_key(value)
    }

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn snapshot(&mut self) -> Snapshot<K, I> {
        Snapshot {
            snapshot: self.relations.snapshot(),
            unify_log_snapshot: self.unify_log.snapshot(),
            modified_snapshot: self.modified_set.snapshot(),
        }
    }

    pub fn rollback_to(&mut self, s: Snapshot<K, I>) {
        let Snapshot { snapshot, unify_log_snapshot, modified_snapshot } = s;
        self.relations.rollback_to(snapshot);
        self.unify_log.rollback_to(unify_log_snapshot);
        self.modified_set.rollback_to(modified_snapshot);
    }

    pub fn commit(&mut self, s: Snapshot<K, I>) {
        let Snapshot { snapshot, unify_log_snapshot, modified_snapshot } = s;
        self.relations.commit(snapshot);
        self.unify_log.commit(unify_log_snapshot);
        self.modified_set.commit(modified_snapshot);
    }

    pub fn vars_since_snapshot(&mut self, s: &Snapshot<K, I>) -> Range<K> {
        self.relations.vars_since_snapshot(&s.snapshot)
    }

    pub fn register(&mut self) -> ms::Offset<I> {
        self.modified_set.register()
    }

    pub fn deregister(&mut self, offset: ms::Offset<I>) {
        self.modified_set.deregister(offset);
    }

    pub fn watch_variable(&mut self, index: I) {
        debug_assert!(index == self.relations.find(index).into());
        self.reference_counts.ensure_contains_elem(index, || 0);
        self.reference_counts[index] += 1;
    }

    pub fn unwatch_variable(&mut self, index: I) {
        self.reference_counts[index] -= 1;
    }

    pub fn drain_modified_set(&mut self, offset: &ms::Offset<I>, mut f: impl FnMut(I) -> bool) {
        let unify_log = &self.unify_log;
        self.modified_set.drain(offset, |vid| {
            for &unified_vid in unify_log.get(vid) {
                f(unified_vid);
            }

            f(vid)
        })
    }
}

pub struct Snapshot<K: ut::UnifyKey, I: Idx = K> {
    snapshot: ut::Snapshot<ut::InPlace<K>>,
    unify_log_snapshot: ul::Snapshot<I>,
    modified_snapshot: ms::Snapshot<I>,
}
