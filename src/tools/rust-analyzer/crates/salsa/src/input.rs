//!
use crate::debug::TableEntry;
use crate::durability::Durability;
use crate::hash::FxIndexMap;
use crate::plumbing::CycleRecoveryStrategy;
use crate::plumbing::InputQueryStorageOps;
use crate::plumbing::QueryStorageMassOps;
use crate::plumbing::QueryStorageOps;
use crate::revision::Revision;
use crate::runtime::StampedValue;
use crate::Database;
use crate::Query;
use crate::Runtime;
use crate::{DatabaseKeyIndex, QueryDb};
use indexmap::map::Entry;
use parking_lot::RwLock;
use std::convert::TryFrom;
use std::iter;
use tracing::debug;

/// Input queries store the result plus a list of the other queries
/// that they invoked. This means we can avoid recomputing them when
/// none of those inputs have changed.
pub struct InputStorage<Q>
where
    Q: Query,
{
    group_index: u16,
    slots: RwLock<FxIndexMap<Q::Key, Slot<Q::Value>>>,
}

struct Slot<V> {
    database_key_index: DatabaseKeyIndex,
    stamped_value: RwLock<StampedValue<V>>,
}

impl<Q> std::panic::RefUnwindSafe for InputStorage<Q>
where
    Q: Query,
    Q::Key: std::panic::RefUnwindSafe,
    Q::Value: std::panic::RefUnwindSafe,
{
}

impl<Q> QueryStorageOps<Q> for InputStorage<Q>
where
    Q: Query,
{
    const CYCLE_STRATEGY: crate::plumbing::CycleRecoveryStrategy = CycleRecoveryStrategy::Panic;

    fn new(group_index: u16) -> Self {
        InputStorage { group_index, slots: Default::default() }
    }

    fn fmt_index(
        &self,
        _db: &<Q as QueryDb<'_>>::DynDb,
        index: DatabaseKeyIndex,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        assert_eq!(index.group_index, self.group_index);
        assert_eq!(index.query_index, Q::QUERY_INDEX);
        let slot_map = self.slots.read();
        let key = slot_map.get_index(index.key_index as usize).unwrap().0;
        write!(fmt, "{}({:?})", Q::QUERY_NAME, key)
    }

    fn maybe_changed_after(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        input: DatabaseKeyIndex,
        revision: Revision,
    ) -> bool {
        assert_eq!(input.group_index, self.group_index);
        assert_eq!(input.query_index, Q::QUERY_INDEX);
        debug_assert!(revision < db.salsa_runtime().current_revision());
        let slots = &self.slots.read();
        let slot = slots.get_index(input.key_index as usize).unwrap().1;

        debug!("maybe_changed_after(slot={:?}, revision={:?})", Q::default(), revision,);

        let changed_at = slot.stamped_value.read().changed_at;

        debug!("maybe_changed_after: changed_at = {:?}", changed_at);

        changed_at > revision
    }

    fn fetch(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Q::Value {
        db.unwind_if_cancelled();

        let slots = &self.slots.read();
        let slot = slots
            .get(key)
            .unwrap_or_else(|| panic!("no value set for {:?}({:?})", Q::default(), key));

        let StampedValue { value, durability, changed_at } = slot.stamped_value.read().clone();

        db.salsa_runtime().report_query_read_and_unwind_if_cycle_resulted(
            slot.database_key_index,
            durability,
            changed_at,
        );

        value
    }

    fn durability(&self, _db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Durability {
        match self.slots.read().get(key) {
            Some(slot) => slot.stamped_value.read().durability,
            None => panic!("no value set for {:?}({:?})", Q::default(), key),
        }
    }

    fn entries<C>(&self, _db: &<Q as QueryDb<'_>>::DynDb) -> C
    where
        C: std::iter::FromIterator<TableEntry<Q::Key, Q::Value>>,
    {
        let slots = self.slots.read();
        slots
            .iter()
            .map(|(key, slot)| {
                TableEntry::new(key.clone(), Some(slot.stamped_value.read().value.clone()))
            })
            .collect()
    }
}

impl<Q> QueryStorageMassOps for InputStorage<Q>
where
    Q: Query,
{
    fn purge(&self) {
        *self.slots.write() = Default::default();
    }
}

impl<Q> InputQueryStorageOps<Q> for InputStorage<Q>
where
    Q: Query,
{
    fn set(&self, runtime: &mut Runtime, key: &Q::Key, value: Q::Value, durability: Durability) {
        tracing::debug!("{:?}({:?}) = {:?} ({:?})", Q::default(), key, value, durability);

        // The value is changing, so we need a new revision (*). We also
        // need to update the 'last changed' revision by invoking
        // `guard.mark_durability_as_changed`.
        //
        // CAREFUL: This will block until the global revision lock can
        // be acquired. If there are still queries executing, they may
        // need to read from this input. Therefore, we wait to acquire
        // the lock on `map` until we also hold the global query write
        // lock.
        //
        // (*) Technically, since you can't presently access an input
        // for a non-existent key, and you can't enumerate the set of
        // keys, we only need a new revision if the key used to
        // exist. But we may add such methods in the future and this
        // case doesn't generally seem worth optimizing for.
        runtime.with_incremented_revision(|next_revision| {
            let mut slots = self.slots.write();

            // Do this *after* we acquire the lock, so that we are not
            // racing with somebody else to modify this same cell.
            // (Otherwise, someone else might write a *newer* revision
            // into the same cell while we block on the lock.)
            let stamped_value = StampedValue { value, durability, changed_at: next_revision };

            match slots.entry(key.clone()) {
                Entry::Occupied(entry) => {
                    let mut slot_stamped_value = entry.get().stamped_value.write();
                    let old_durability = slot_stamped_value.durability;
                    *slot_stamped_value = stamped_value;
                    Some(old_durability)
                }

                Entry::Vacant(entry) => {
                    let key_index = u32::try_from(entry.index()).unwrap();
                    let database_key_index = DatabaseKeyIndex {
                        group_index: self.group_index,
                        query_index: Q::QUERY_INDEX,
                        key_index,
                    };
                    entry.insert(Slot {
                        database_key_index,
                        stamped_value: RwLock::new(stamped_value),
                    });
                    None
                }
            }
        });
    }
}

/// Same as `InputStorage`, but optimized for queries that take no inputs.
pub struct UnitInputStorage<Q>
where
    Q: Query<Key = ()>,
{
    group_index: u16,
    slot: UnitSlot<Q::Value>,
}

struct UnitSlot<V> {
    database_key_index: DatabaseKeyIndex,
    stamped_value: RwLock<Option<StampedValue<V>>>,
}

impl<Q> std::panic::RefUnwindSafe for UnitInputStorage<Q>
where
    Q: Query<Key = ()>,
    Q::Key: std::panic::RefUnwindSafe,
    Q::Value: std::panic::RefUnwindSafe,
{
}

impl<Q> QueryStorageOps<Q> for UnitInputStorage<Q>
where
    Q: Query<Key = ()>,
{
    const CYCLE_STRATEGY: crate::plumbing::CycleRecoveryStrategy = CycleRecoveryStrategy::Panic;

    fn new(group_index: u16) -> Self {
        let database_key_index =
            DatabaseKeyIndex { group_index, query_index: Q::QUERY_INDEX, key_index: 0 };
        UnitInputStorage {
            group_index,
            slot: UnitSlot { database_key_index, stamped_value: RwLock::new(None) },
        }
    }

    fn fmt_index(
        &self,
        _db: &<Q as QueryDb<'_>>::DynDb,
        index: DatabaseKeyIndex,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        assert_eq!(index.group_index, self.group_index);
        assert_eq!(index.query_index, Q::QUERY_INDEX);
        write!(fmt, "{}", Q::QUERY_NAME)
    }

    fn maybe_changed_after(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        input: DatabaseKeyIndex,
        revision: Revision,
    ) -> bool {
        assert_eq!(input.group_index, self.group_index);
        assert_eq!(input.query_index, Q::QUERY_INDEX);
        debug_assert!(revision < db.salsa_runtime().current_revision());

        debug!("maybe_changed_after(slot={:?}, revision={:?})", Q::default(), revision,);

        let changed_at = self.slot.stamped_value.read().as_ref().unwrap().changed_at;

        debug!("maybe_changed_after: changed_at = {:?}", changed_at);

        changed_at > revision
    }

    fn fetch(&self, db: &<Q as QueryDb<'_>>::DynDb, &(): &Q::Key) -> Q::Value {
        db.unwind_if_cancelled();

        let StampedValue { value, durability, changed_at } = self
            .slot
            .stamped_value
            .read()
            .clone()
            .unwrap_or_else(|| panic!("no value set for {:?}", Q::default()));

        db.salsa_runtime().report_query_read_and_unwind_if_cycle_resulted(
            self.slot.database_key_index,
            durability,
            changed_at,
        );

        value
    }

    fn durability(&self, _db: &<Q as QueryDb<'_>>::DynDb, &(): &Q::Key) -> Durability {
        match &*self.slot.stamped_value.read() {
            Some(stamped_value) => stamped_value.durability,
            None => panic!("no value set for {:?}", Q::default(),),
        }
    }

    fn entries<C>(&self, _db: &<Q as QueryDb<'_>>::DynDb) -> C
    where
        C: std::iter::FromIterator<TableEntry<Q::Key, Q::Value>>,
    {
        iter::once(TableEntry::new(
            (),
            self.slot.stamped_value.read().as_ref().map(|it| it.value.clone()),
        ))
        .collect()
    }
}

impl<Q> QueryStorageMassOps for UnitInputStorage<Q>
where
    Q: Query<Key = ()>,
{
    fn purge(&self) {
        *self.slot.stamped_value.write() = Default::default();
    }
}

impl<Q> InputQueryStorageOps<Q> for UnitInputStorage<Q>
where
    Q: Query<Key = ()>,
{
    fn set(&self, runtime: &mut Runtime, (): &Q::Key, value: Q::Value, durability: Durability) {
        tracing::debug!("{:?} = {:?} ({:?})", Q::default(), value, durability);

        // The value is changing, so we need a new revision (*). We also
        // need to update the 'last changed' revision by invoking
        // `guard.mark_durability_as_changed`.
        //
        // CAREFUL: This will block until the global revision lock can
        // be acquired. If there are still queries executing, they may
        // need to read from this input. Therefore, we wait to acquire
        // the lock on `map` until we also hold the global query write
        // lock.
        //
        // (*) Technically, since you can't presently access an input
        // for a non-existent key, and you can't enumerate the set of
        // keys, we only need a new revision if the key used to
        // exist. But we may add such methods in the future and this
        // case doesn't generally seem worth optimizing for.
        runtime.with_incremented_revision(|next_revision| {
            let mut stamped_value_slot = self.slot.stamped_value.write();

            // Do this *after* we acquire the lock, so that we are not
            // racing with somebody else to modify this same cell.
            // (Otherwise, someone else might write a *newer* revision
            // into the same cell while we block on the lock.)
            let stamped_value = StampedValue { value, durability, changed_at: next_revision };

            match &mut *stamped_value_slot {
                Some(slot_stamped_value) => {
                    let old_durability = slot_stamped_value.durability;
                    *slot_stamped_value = stamped_value;
                    Some(old_durability)
                }

                stamped_value_slot @ None => {
                    *stamped_value_slot = Some(stamped_value);
                    None
                }
            }
        });
    }
}

/// Check that `Slot<Q, MP>: Send + Sync` as long as
/// `DB::DatabaseData: Send + Sync`, which in turn implies that
/// `Q::Key: Send + Sync`, `Q::Value: Send + Sync`.
#[allow(dead_code)]
fn check_send_sync<Q>()
where
    Q: Query,
    Q::Key: Send + Sync,
    Q::Value: Send + Sync,
{
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<Slot<Q::Value>>();
    is_send_sync::<UnitSlot<Q::Value>>();
}

/// Check that `Slot<Q, MP>: 'static` as long as
/// `DB::DatabaseData: 'static`, which in turn implies that
/// `Q::Key: 'static`, `Q::Value: 'static`.
#[allow(dead_code)]
fn check_static<Q>()
where
    Q: Query + 'static,
    Q::Key: 'static,
    Q::Value: 'static,
{
    fn is_static<T: 'static>() {}
    is_static::<Slot<Q::Value>>();
    is_static::<UnitSlot<Q::Value>>();
}
