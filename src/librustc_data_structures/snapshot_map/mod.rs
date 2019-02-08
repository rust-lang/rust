use crate::fx::FxHashMap;
use std::hash::Hash;
use std::ops;
use std::mem;

#[cfg(test)]
mod test;

pub struct SnapshotMap<K, V>
    where K: Hash + Clone + Eq
{
    map: FxHashMap<K, V>,
    undo_log: Vec<UndoLog<K, V>>,
    num_open_snapshots: usize,
}

// HACK(eddyb) manual impl avoids `Default` bounds on `K` and `V`.
impl<K, V> Default for SnapshotMap<K, V>
    where K: Hash + Clone + Eq
{
    fn default() -> Self {
        SnapshotMap {
            map: Default::default(),
            undo_log: Default::default(),
            num_open_snapshots: 0,
        }
    }
}

pub struct Snapshot {
    len: usize,
}

enum UndoLog<K, V> {
    Inserted(K),
    Overwrite(K, V),
    Purged,
}

impl<K, V> SnapshotMap<K, V>
    where K: Hash + Clone + Eq
{
    pub fn clear(&mut self) {
        self.map.clear();
        self.undo_log.clear();
        self.num_open_snapshots = 0;
    }

    fn in_snapshot(&self) -> bool {
        self.num_open_snapshots > 0
    }

    pub fn insert(&mut self, key: K, value: V) -> bool {
        match self.map.insert(key.clone(), value) {
            None => {
                if self.in_snapshot() {
                    self.undo_log.push(UndoLog::Inserted(key));
                }
                true
            }
            Some(old_value) => {
                if self.in_snapshot() {
                    self.undo_log.push(UndoLog::Overwrite(key, old_value));
                }
                false
            }
        }
    }

    pub fn remove(&mut self, key: K) -> bool {
        match self.map.remove(&key) {
            Some(old_value) => {
                if self.in_snapshot() {
                    self.undo_log.push(UndoLog::Overwrite(key, old_value));
                }
                true
            }
            None => false,
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    pub fn snapshot(&mut self) -> Snapshot {
        let len = self.undo_log.len();
        self.num_open_snapshots += 1;
        Snapshot { len }
    }

    fn assert_open_snapshot(&self, snapshot: &Snapshot) {
        assert!(self.undo_log.len() >= snapshot.len);
        assert!(self.num_open_snapshots > 0);
    }

    pub fn commit(&mut self, snapshot: Snapshot) {
        self.assert_open_snapshot(&snapshot);
        if self.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.len == 0);
            self.undo_log.clear();
        }

        self.num_open_snapshots -= 1;
    }

    pub fn partial_rollback<F>(&mut self,
                               snapshot: &Snapshot,
                               should_revert_key: &F)
        where F: Fn(&K) -> bool
    {
        self.assert_open_snapshot(snapshot);
        for i in (snapshot.len .. self.undo_log.len()).rev() {
            let reverse = match self.undo_log[i] {
                UndoLog::Purged => false,
                UndoLog::Inserted(ref k) => should_revert_key(k),
                UndoLog::Overwrite(ref k, _) => should_revert_key(k),
            };

            if reverse {
                let entry = mem::replace(&mut self.undo_log[i], UndoLog::Purged);
                self.reverse(entry);
            }
        }
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot) {
        self.assert_open_snapshot(&snapshot);
        while self.undo_log.len() > snapshot.len {
            let entry = self.undo_log.pop().unwrap();
            self.reverse(entry);
        }

        self.num_open_snapshots -= 1;
    }

    fn reverse(&mut self, entry: UndoLog<K, V>) {
        match entry {
            UndoLog::Inserted(key) => {
                self.map.remove(&key);
            }

            UndoLog::Overwrite(key, old_value) => {
                self.map.insert(key, old_value);
            }

            UndoLog::Purged => {}
        }
    }
}

impl<'k, K, V> ops::Index<&'k K> for SnapshotMap<K, V>
    where K: Hash + Clone + Eq
{
    type Output = V;
    fn index(&self, key: &'k K) -> &V {
        &self.map[key]
    }
}
