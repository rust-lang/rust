// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A utility class for implementing "snapshottable" things; a snapshottable data structure permits
//! you to take a snapshot (via `start_snapshot`) and then, after making some changes, elect either
//! to rollback to the start of the snapshot or commit those changes.
//!
//! This vector is intended to be used as part of an abstraction, not serve as a complete
//! abstraction on its own. As such, while it will roll back most changes on its own, it also
//! supports a `get_mut` operation that gives you an arbitrary mutable pointer into the vector. To
//! ensure that any changes you make this with this pointer are rolled back, you must invoke
//! `record` to record any changes you make and also supplying a delegate capable of reversing
//! those changes.
use self::UndoLog::*;

use std::mem;
use std::ops;

pub enum UndoLog<D: SnapshotVecDelegate> {
    /// Indicates where a snapshot started.
    OpenSnapshot,

    /// Indicates a snapshot that has been committed.
    CommittedSnapshot,

    /// New variable with given index was created.
    NewElem(usize),

    /// Variable with given index was changed *from* the given value.
    SetElem(usize, D::Value),

    /// Extensible set of actions
    Other(D::Undo),
}

pub struct SnapshotVec<D: SnapshotVecDelegate> {
    values: Vec<D::Value>,
    undo_log: Vec<UndoLog<D>>,
}

// Snapshots are tokens that should be created/consumed linearly.
pub struct Snapshot {
    // Length of the undo log at the time the snapshot was taken.
    length: usize,
}

pub trait SnapshotVecDelegate {
    type Value;
    type Undo;

    fn reverse(values: &mut Vec<Self::Value>, action: Self::Undo);
}

impl<D: SnapshotVecDelegate> SnapshotVec<D> {
    pub fn new() -> SnapshotVec<D> {
        SnapshotVec {
            values: Vec::new(),
            undo_log: Vec::new(),
        }
    }

    fn in_snapshot(&self) -> bool {
        !self.undo_log.is_empty()
    }

    pub fn record(&mut self, action: D::Undo) {
        if self.in_snapshot() {
            self.undo_log.push(Other(action));
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn push(&mut self, elem: D::Value) -> usize {
        let len = self.values.len();
        self.values.push(elem);

        if self.in_snapshot() {
            self.undo_log.push(NewElem(len));
        }

        len
    }

    pub fn get(&self, index: usize) -> &D::Value {
        &self.values[index]
    }

    /// Returns a mutable pointer into the vec; whatever changes you make here cannot be undone
    /// automatically, so you should be sure call `record()` with some sort of suitable undo
    /// action.
    pub fn get_mut(&mut self, index: usize) -> &mut D::Value {
        &mut self.values[index]
    }

    /// Updates the element at the given index. The old value will saved (and perhaps restored) if
    /// a snapshot is active.
    pub fn set(&mut self, index: usize, new_elem: D::Value) {
        let old_elem = mem::replace(&mut self.values[index], new_elem);
        if self.in_snapshot() {
            self.undo_log.push(SetElem(index, old_elem));
        }
    }

    pub fn start_snapshot(&mut self) -> Snapshot {
        let length = self.undo_log.len();
        self.undo_log.push(OpenSnapshot);
        Snapshot { length: length }
    }

    pub fn actions_since_snapshot(&self, snapshot: &Snapshot) -> &[UndoLog<D>] {
        &self.undo_log[snapshot.length..]
    }

    fn assert_open_snapshot(&self, snapshot: &Snapshot) {
        // Or else there was a failure to follow a stack discipline:
        assert!(self.undo_log.len() > snapshot.length);

        // Invariant established by start_snapshot():
        assert!(match self.undo_log[snapshot.length] {
            OpenSnapshot => true,
            _ => false,
        });
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot) {
        debug!("rollback_to({})", snapshot.length);

        self.assert_open_snapshot(&snapshot);

        while self.undo_log.len() > snapshot.length + 1 {
            match self.undo_log.pop().unwrap() {
                OpenSnapshot => {
                    // This indicates a failure to obey the stack discipline.
                    panic!("Cannot rollback an uncommitted snapshot");
                }

                CommittedSnapshot => {
                    // This occurs when there are nested snapshots and
                    // the inner is committed but outer is rolled back.
                }

                NewElem(i) => {
                    self.values.pop();
                    assert!(self.values.len() == i);
                }

                SetElem(i, v) => {
                    self.values[i] = v;
                }

                Other(u) => {
                    D::reverse(&mut self.values, u);
                }
            }
        }

        let v = self.undo_log.pop().unwrap();
        assert!(match v {
            OpenSnapshot => true,
            _ => false,
        });
        assert!(self.undo_log.len() == snapshot.length);
    }

    /// Commits all changes since the last snapshot. Of course, they
    /// can still be undone if there is a snapshot further out.
    pub fn commit(&mut self, snapshot: Snapshot) {
        debug!("commit({})", snapshot.length);

        self.assert_open_snapshot(&snapshot);

        if snapshot.length == 0 {
            // The root snapshot.
            self.undo_log.truncate(0);
        } else {
            self.undo_log[snapshot.length] = CommittedSnapshot;
        }
    }
}

impl<D: SnapshotVecDelegate> ops::Deref for SnapshotVec<D> {
    type Target = [D::Value];
    fn deref(&self) -> &[D::Value] {
        &*self.values
    }
}

impl<D: SnapshotVecDelegate> ops::DerefMut for SnapshotVec<D> {
    fn deref_mut(&mut self) -> &mut [D::Value] {
        &mut *self.values
    }
}

impl<D: SnapshotVecDelegate> ops::Index<usize> for SnapshotVec<D> {
    type Output = D::Value;
    fn index(&self, index: usize) -> &D::Value {
        self.get(index)
    }
}

impl<D: SnapshotVecDelegate> ops::IndexMut<usize> for SnapshotVec<D> {
    fn index_mut(&mut self, index: usize) -> &mut D::Value {
        self.get_mut(index)
    }
}

impl<D: SnapshotVecDelegate> Extend<D::Value> for SnapshotVec<D> {
    fn extend<T>(&mut self, iterable: T) where T: IntoIterator<Item=D::Value> {
        for item in iterable {
            self.push(item);
        }
    }
}
