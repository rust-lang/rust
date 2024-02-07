use crate::debug::TableEntry;
use crate::durability::Durability;
use crate::intern_id::InternId;
use crate::plumbing::CycleRecoveryStrategy;
use crate::plumbing::HasQueryGroup;
use crate::plumbing::QueryStorageMassOps;
use crate::plumbing::QueryStorageOps;
use crate::revision::Revision;
use crate::Query;
use crate::{Database, DatabaseKeyIndex, QueryDb};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::convert::From;
use std::fmt::Debug;
use std::hash::Hash;
use triomphe::Arc;

const INTERN_DURABILITY: Durability = Durability::HIGH;

/// Handles storage where the value is 'derived' by executing a
/// function (in contrast to "inputs").
pub struct InternedStorage<Q>
where
    Q: Query,
    Q::Value: InternKey,
{
    group_index: u16,
    tables: RwLock<InternTables<Q::Key>>,
}

/// Storage for the looking up interned things.
pub struct LookupInternedStorage<Q, IQ>
where
    Q: Query,
    Q::Key: InternKey,
    Q::Value: Eq + Hash,
{
    phantom: std::marker::PhantomData<(Q::Key, IQ)>,
}

struct InternTables<K> {
    /// Map from the key to the corresponding intern-index.
    map: FxHashMap<K, InternId>,

    /// For each valid intern-index, stores the interned value.
    values: Vec<Arc<Slot<K>>>,
}

/// Trait implemented for the "key" that results from a
/// `#[salsa::intern]` query.  This is basically meant to be a
/// "newtype"'d `u32`.
pub trait InternKey {
    /// Create an instance of the intern-key from a `u32` value.
    fn from_intern_id(v: InternId) -> Self;

    /// Extract the `u32` with which the intern-key was created.
    fn as_intern_id(&self) -> InternId;
}

impl InternKey for InternId {
    fn from_intern_id(v: InternId) -> InternId {
        v
    }

    fn as_intern_id(&self) -> InternId {
        *self
    }
}

#[derive(Debug)]
struct Slot<K> {
    /// DatabaseKeyIndex for this slot.
    database_key_index: DatabaseKeyIndex,

    /// Value that was interned.
    value: K,

    /// When was this intern'd?
    ///
    /// (This informs the "changed-at" result)
    interned_at: Revision,
}

impl<Q> std::panic::RefUnwindSafe for InternedStorage<Q>
where
    Q: Query,
    Q::Key: std::panic::RefUnwindSafe,
    Q::Value: InternKey,
    Q::Value: std::panic::RefUnwindSafe,
{
}

impl<K: Debug + Hash + Eq> InternTables<K> {
    /// Returns the slot for the given key.
    fn slot_for_key(&self, key: &K) -> Option<(Arc<Slot<K>>, InternId)> {
        let &index = self.map.get(key)?;
        Some((self.slot_for_index(index), index))
    }

    /// Returns the slot at the given index.
    fn slot_for_index(&self, index: InternId) -> Arc<Slot<K>> {
        let slot = &self.values[index.as_usize()];
        slot.clone()
    }
}

impl<K> Default for InternTables<K>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self {
            map: Default::default(),
            values: Default::default(),
        }
    }
}

impl<Q> InternedStorage<Q>
where
    Q: Query,
    Q::Key: Eq + Hash + Clone,
    Q::Value: InternKey,
{
    /// If `key` has already been interned, returns its slot. Otherwise, creates a new slot.
    fn intern_index(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        key: &Q::Key,
    ) -> (Arc<Slot<Q::Key>>, InternId) {
        if let Some(i) = self.intern_check(key) {
            return i;
        }

        let owned_key1 = key.to_owned();
        let owned_key2 = owned_key1.clone();
        let revision_now = db.salsa_runtime().current_revision();

        let mut tables = self.tables.write();
        let tables = &mut *tables;
        let entry = match tables.map.entry(owned_key1) {
            Entry::Vacant(entry) => entry,
            Entry::Occupied(entry) => {
                // Somebody inserted this key while we were waiting
                // for the write lock. In this case, we don't need to
                // update the `accessed_at` field because they should
                // have already done so!
                let index = *entry.get();
                let slot = &tables.values[index.as_usize()];
                debug_assert_eq!(owned_key2, slot.value);
                return (slot.clone(), index);
            }
        };

        let create_slot = |index: InternId| {
            let database_key_index = DatabaseKeyIndex {
                group_index: self.group_index,
                query_index: Q::QUERY_INDEX,
                key_index: index.as_u32(),
            };
            Arc::new(Slot {
                database_key_index,
                value: owned_key2,
                interned_at: revision_now,
            })
        };

        let (slot, index);
        index = InternId::from(tables.values.len());
        slot = create_slot(index);
        tables.values.push(slot.clone());
        entry.insert(index);

        (slot, index)
    }

    fn intern_check(&self, key: &Q::Key) -> Option<(Arc<Slot<Q::Key>>, InternId)> {
        self.tables.read().slot_for_key(key)
    }

    /// Given an index, lookup and clone its value, updating the
    /// `accessed_at` time if necessary.
    fn lookup_value(&self, index: InternId) -> Arc<Slot<Q::Key>> {
        self.tables.read().slot_for_index(index)
    }
}

impl<Q> QueryStorageOps<Q> for InternedStorage<Q>
where
    Q: Query,
    Q::Value: InternKey,
{
    const CYCLE_STRATEGY: crate::plumbing::CycleRecoveryStrategy = CycleRecoveryStrategy::Panic;

    fn new(group_index: u16) -> Self {
        InternedStorage {
            group_index,
            tables: RwLock::new(InternTables::default()),
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
        let intern_id = InternId::from(index.key_index);
        let slot = self.lookup_value(intern_id);
        write!(fmt, "{}({:?})", Q::QUERY_NAME, slot.value)
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
        let intern_id = InternId::from(input.key_index);
        let slot = self.lookup_value(intern_id);
        slot.maybe_changed_after(revision)
    }

    fn fetch(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Q::Value {
        db.unwind_if_cancelled();
        let (slot, index) = self.intern_index(db, key);
        let changed_at = slot.interned_at;
        db.salsa_runtime()
            .report_query_read_and_unwind_if_cycle_resulted(
                slot.database_key_index,
                INTERN_DURABILITY,
                changed_at,
            );
        <Q::Value>::from_intern_id(index)
    }

    fn durability(&self, _db: &<Q as QueryDb<'_>>::DynDb, _key: &Q::Key) -> Durability {
        INTERN_DURABILITY
    }

    fn entries<C>(&self, _db: &<Q as QueryDb<'_>>::DynDb) -> C
    where
        C: std::iter::FromIterator<TableEntry<Q::Key, Q::Value>>,
    {
        let tables = self.tables.read();
        tables
            .map
            .iter()
            .map(|(key, index)| {
                TableEntry::new(key.clone(), Some(<Q::Value>::from_intern_id(*index)))
            })
            .collect()
    }
}

impl<Q> QueryStorageMassOps for InternedStorage<Q>
where
    Q: Query,
    Q::Value: InternKey,
{
    fn purge(&self) {
        *self.tables.write() = Default::default();
    }
}

// Workaround for
// ```
// IQ: for<'d> QueryDb<
//     'd,
//     DynDb = <Q as QueryDb<'d>>::DynDb,
//     Group = <Q as QueryDb<'d>>::Group,
//     GroupStorage = <Q as QueryDb<'d>>::GroupStorage,
// >,
// ```
// not working to make rustc know DynDb, Group and GroupStorage being the same in `Q` and `IQ`
#[doc(hidden)]
pub trait EqualDynDb<'d, IQ>: QueryDb<'d>
where
    IQ: QueryDb<'d>,
{
    fn convert_db(d: &Self::DynDb) -> &IQ::DynDb;
    fn convert_group_storage(d: &Self::GroupStorage) -> &IQ::GroupStorage;
}

impl<'d, IQ, Q> EqualDynDb<'d, IQ> for Q
where
    Q: QueryDb<'d, DynDb = IQ::DynDb, Group = IQ::Group, GroupStorage = IQ::GroupStorage>,
    Q::DynDb: HasQueryGroup<Q::Group>,
    IQ: QueryDb<'d>,
{
    fn convert_db(d: &Self::DynDb) -> &IQ::DynDb {
        d
    }
    fn convert_group_storage(d: &Self::GroupStorage) -> &IQ::GroupStorage {
        d
    }
}

impl<Q, IQ> QueryStorageOps<Q> for LookupInternedStorage<Q, IQ>
where
    Q: Query,
    Q::Key: InternKey,
    Q::Value: Eq + Hash,
    IQ: Query<Key = Q::Value, Value = Q::Key, Storage = InternedStorage<IQ>>,
    for<'d> Q: EqualDynDb<'d, IQ>,
{
    const CYCLE_STRATEGY: CycleRecoveryStrategy = CycleRecoveryStrategy::Panic;

    fn new(_group_index: u16) -> Self {
        LookupInternedStorage {
            phantom: std::marker::PhantomData,
        }
    }

    fn fmt_index(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        index: DatabaseKeyIndex,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let group_storage =
            <<Q as QueryDb<'_>>::DynDb as HasQueryGroup<Q::Group>>::group_storage(db);
        let interned_storage = IQ::query_storage(Q::convert_group_storage(group_storage));
        interned_storage.fmt_index(Q::convert_db(db), index, fmt)
    }

    fn maybe_changed_after(
        &self,
        db: &<Q as QueryDb<'_>>::DynDb,
        input: DatabaseKeyIndex,
        revision: Revision,
    ) -> bool {
        let group_storage =
            <<Q as QueryDb<'_>>::DynDb as HasQueryGroup<Q::Group>>::group_storage(db);
        let interned_storage = IQ::query_storage(Q::convert_group_storage(group_storage));
        interned_storage.maybe_changed_after(Q::convert_db(db), input, revision)
    }

    fn fetch(&self, db: &<Q as QueryDb<'_>>::DynDb, key: &Q::Key) -> Q::Value {
        let index = key.as_intern_id();
        let group_storage =
            <<Q as QueryDb<'_>>::DynDb as HasQueryGroup<Q::Group>>::group_storage(db);
        let interned_storage = IQ::query_storage(Q::convert_group_storage(group_storage));
        let slot = interned_storage.lookup_value(index);
        let value = slot.value.clone();
        let interned_at = slot.interned_at;
        db.salsa_runtime()
            .report_query_read_and_unwind_if_cycle_resulted(
                slot.database_key_index,
                INTERN_DURABILITY,
                interned_at,
            );
        value
    }

    fn durability(&self, _db: &<Q as QueryDb<'_>>::DynDb, _key: &Q::Key) -> Durability {
        INTERN_DURABILITY
    }

    fn entries<C>(&self, db: &<Q as QueryDb<'_>>::DynDb) -> C
    where
        C: std::iter::FromIterator<TableEntry<Q::Key, Q::Value>>,
    {
        let group_storage =
            <<Q as QueryDb<'_>>::DynDb as HasQueryGroup<Q::Group>>::group_storage(db);
        let interned_storage = IQ::query_storage(Q::convert_group_storage(group_storage));
        let tables = interned_storage.tables.read();
        tables
            .map
            .iter()
            .map(|(key, index)| {
                TableEntry::new(<Q::Key>::from_intern_id(*index), Some(key.clone()))
            })
            .collect()
    }
}

impl<Q, IQ> QueryStorageMassOps for LookupInternedStorage<Q, IQ>
where
    Q: Query,
    Q::Key: InternKey,
    Q::Value: Eq + Hash,
    IQ: Query<Key = Q::Value, Value = Q::Key>,
{
    fn purge(&self) {}
}

impl<K> Slot<K> {
    fn maybe_changed_after(&self, revision: Revision) -> bool {
        self.interned_at > revision
    }
}

/// Check that `Slot<Q, MP>: Send + Sync` as long as
/// `DB::DatabaseData: Send + Sync`, which in turn implies that
/// `Q::Key: Send + Sync`, `Q::Value: Send + Sync`.
#[allow(dead_code)]
fn check_send_sync<K>()
where
    K: Send + Sync,
{
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<Slot<K>>();
}

/// Check that `Slot<Q, MP>: 'static` as long as
/// `DB::DatabaseData: 'static`, which in turn implies that
/// `Q::Key: 'static`, `Q::Value: 'static`.
#[allow(dead_code)]
fn check_static<K>()
where
    K: 'static,
{
    fn is_static<T: 'static>() {}
    is_static::<Slot<K>>();
}
