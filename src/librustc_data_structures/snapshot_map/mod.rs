// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fnv::FnvHashMap;
use std::hash::Hash;
use std::ops;

#[cfg(test)]
mod test;

pub struct SnapshotMap<K, V>
    where K: Hash + Clone + Eq
{
    map: FnvHashMap<K, V>,
    undo_log: Vec<UndoLog<K, V>>,
}

pub struct Snapshot {
    len: usize
}

enum UndoLog<K, V> {
    OpenSnapshot,
    CommittedSnapshot,
    Inserted(K),
    Overwrite(K, V),
}

impl<K, V> SnapshotMap<K, V>
    where K: Hash + Clone + Eq
{
    pub fn new() -> Self {
        SnapshotMap {
            map: FnvHashMap(),
            undo_log: vec![]
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> bool {
        match self.map.insert(key.clone(), value) {
            None => {
                if !self.undo_log.is_empty() {
                    self.undo_log.push(UndoLog::Inserted(key));
                }
                true
            }
            Some(old_value) => {
                if !self.undo_log.is_empty() {
                    self.undo_log.push(UndoLog::Overwrite(key, old_value));
                }
                false
            }
        }
    }

    pub fn remove(&mut self, key: K) -> bool {
        match self.map.remove(&key) {
            Some(old_value) => {
                if !self.undo_log.is_empty() {
                    self.undo_log.push(UndoLog::Overwrite(key, old_value));
                }
                true
            }
            None => {
                false
            }
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    pub fn snapshot(&mut self) -> Snapshot {
        self.undo_log.push(UndoLog::OpenSnapshot);
        let len = self.undo_log.len() - 1;
        Snapshot { len: len }
    }

    fn assert_open_snapshot(&self, snapshot: &Snapshot) {
        assert!(snapshot.len < self.undo_log.len());
        assert!(match self.undo_log[snapshot.len] {
            UndoLog::OpenSnapshot => true,
            _ => false
        });
    }

    pub fn commit(&mut self, snapshot: Snapshot) {
        self.assert_open_snapshot(&snapshot);
        if snapshot.len == 0 {
            // The root snapshot.
            self.undo_log.truncate(0);
        } else {
            self.undo_log[snapshot.len] = UndoLog::CommittedSnapshot;
        }
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot) {
        self.assert_open_snapshot(&snapshot);
        while self.undo_log.len() > snapshot.len + 1 {
            match self.undo_log.pop().unwrap() {
                UndoLog::OpenSnapshot => {
                    panic!("cannot rollback an uncommitted snapshot");
                }

                UndoLog::CommittedSnapshot => { }

                UndoLog::Inserted(key) => {
                    self.map.remove(&key);
                }

                UndoLog::Overwrite(key, old_value) => {
                    self.map.insert(key, old_value);
                }
            }
        }

        let v = self.undo_log.pop().unwrap();
        assert!(match v { UndoLog::OpenSnapshot => true, _ => false });
        assert!(self.undo_log.len() == snapshot.len);
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
