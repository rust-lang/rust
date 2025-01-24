use crate::debug::TableEntry;
use crate::durability::Durability;
use crate::hash::FxIndexMap;
use crate::plumbing::DerivedQueryStorageOps;
use crate::plumbing::QueryFunction;
use crate::plumbing::QueryStorageMassOps;
use crate::plumbing::QueryStorageOps;
use crate::runtime::StampedValue;
use crate::Runtime;
use crate::{Database, DatabaseKeyIndex, QueryDb, Revision};
use parking_lot::RwLock;
use std::borrow::Borrow;
use std::hash::Hash;
use triomphe::Arc;

mod slot;
use slot::Slot;

/// Memoized queries store the result plus a list of the other queries
/// that they invoked. This means we can avoid recomputing them when
/// none of those inputs have changed.
pub type MemoizedStorage<Q> = DerivedStorage<Q>;

/// Handles storage where the value is 'derived' by executing a
/// function (in contrast to "inputs").
pub struct DerivedStorage<Q>
where
    Q: QueryFunction,
{
    group_index: u16,
    slot_map: RwLock<FxIndexMap<Q::Key, Arc<Slot<Q>>>>,
}

impl<Q> std::panic::RefUnwindSafe for DerivedStorage<Q>
where
    Q: QueryFunction,

    Q::Key: std::panic::RefUnwindSafe,
    Q::Value: std::panic::RefUnwindSafe,
{
}

impl<Q> DerivedStorage<Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    fn slot(&self, key: &Q::Key) -> Arc<Slot<Q>> {
        if let Some(v) = self.slot_map.read().get(key) {
            return v.clone();
        }

        let mut write = self.slot_map.write();
        let entry = write.entry(key.clone());
        let key_index = entry.index() as u32;
        let database_key_index = DatabaseKeyIndex {
            group_index: self.group_index,
            query_index: Q::QUERY_INDEX,
            key_index,
        };
        entry.or_insert_with(|| Arc::new(Slot::new(database_key_index))).clone()
    }
}

impl<Q> QueryStorageOps<Q> for DerivedStorage<Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    const CYCLE_STRATEGY: crate::plumbing::CycleRecoveryStrategy = Q::CYCLE_STRATEGY;

    fn new(group_index: u16) -> Self {
        DerivedStorage { group_index, slot_map: RwLock::new(FxIndexMap::default()) }
    }

    fn fmt_index(
        &self,
        _db: &<Q as QueryDb<'_>>::DynDb,
        index: u32,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let slot_map = self.slot_map.read();
        let key = slot_map.get_index(index as usize).unwrap().0;
        write!(fmt, "{}::{}({:?})", std::any::type_name::<Q>(), Q::QUERY_NAME, key)
    }

    fn maybe_changed_after(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        index: u32,
        revision: Revision,
    ) -> bool {
        debug_assert!(revision < db.salsa_runtime().current_revision());
        let (key, slot) = {
            let read = self.slot_map.read();
            let Some((key, slot)) = read.get_index(index as usize) else {
                return false;
            };
            (key.clone(), slot.clone())
        };
        slot.maybe_changed_after(db, revision, &key)
    }

    fn fetch(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Q::Value {
        db.unwind_if_cancelled();

        let slot = self.slot(key);
        let StampedValue { value, durability, changed_at } = slot.read(db, key);

        db.salsa_runtime().report_query_read_and_unwind_if_cycle_resulted(
            slot.database_key_index(),
            durability,
            changed_at,
        );

        value
    }

    fn durability(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Durability {
        self.slot_map.read().get(key).map_or(Durability::LOW, |slot| slot.durability(db))
    }

    fn entries<C>(&self, _db: &<Q as QueryDb<'_>>::DynDb) -> C
    where
        C: std::iter::FromIterator<TableEntry<Q::Key, Q::Value>>,
    {
        let slot_map = self.slot_map.read();
        slot_map.iter().filter_map(|(key, slot)| slot.as_table_entry(key)).collect()
    }
}

impl<Q> QueryStorageMassOps for DerivedStorage<Q>
where
    Q: QueryFunction,
{
    fn purge(&self) {
        *self.slot_map.write() = Default::default();
    }
}

impl<Q> DerivedQueryStorageOps<Q> for DerivedStorage<Q>
where
    Q: QueryFunction,
    Q::Value: Eq,
{
    fn invalidate<S>(&self, runtime: &mut Runtime, key: &S)
    where
        S: Eq + Hash,
        Q::Key: Borrow<S>,
    {
        runtime.with_incremented_revision(|new_revision| {
            let map_read = self.slot_map.read();

            if let Some(slot) = map_read.get(key) {
                if let Some(durability) = slot.invalidate(new_revision) {
                    return Some(durability);
                }
            }

            None
        })
    }
}
