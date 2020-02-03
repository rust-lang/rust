use std::{collections::VecDeque, marker::PhantomData};

use rustc_index::{bit_set::BitSet, vec::Idx};

#[derive(Clone, Debug)]
pub struct ModifiedSet<T: Idx> {
    modified: VecDeque<T>,
    snapshots: usize,
    modified_set: BitSet<T>,
    undo_offsets: Vec<usize>,
    offsets: Vec<usize>,
}

impl<T: Idx> Default for ModifiedSet<T> {
    fn default() -> Self {
        Self {
            modified: Default::default(),
            snapshots: 0,
            modified_set: BitSet::new_empty(0),
            offsets: Vec::new(),
            undo_offsets: Vec::new(),
        }
    }
}

impl<T: Idx> ModifiedSet<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, index: T) {
        if index.index() >= self.modified_set.domain_size() {
            self.modified_set.resize(index.index() + 1);
        }
        if self.modified_set.insert(index) {
            self.modified.push_back(index);
        }
    }

    pub fn drain(&mut self, offset: &Offset<T>, mut f: impl FnMut(T) -> bool) {
        let offset = &mut self.offsets[offset.index];
        for &index in self.modified.iter().skip(*offset) {
            if f(index) {}
        }
        *offset = self.modified.len();
    }

    pub fn snapshot(&mut self) -> Snapshot<T> {
        self.snapshots += 1;
        let offsets_start = self.undo_offsets.len();
        self.undo_offsets.extend_from_slice(&self.offsets);
        Snapshot {
            modified_len: self.modified.len(),
            offsets_start,
            offsets_len: self.offsets.len(),
            _marker: PhantomData,
        }
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot<T>) {
        self.snapshots -= 1;
        for &index in self.modified.iter().skip(snapshot.modified_len) {
            self.modified_set.remove(index);
        }
        self.modified.truncate(snapshot.modified_len);
        let mut offsets = self.offsets.iter_mut();
        for (offset, &saved_offset) in offsets.by_ref().zip(
            &self.undo_offsets
                [snapshot.offsets_start..snapshot.offsets_start + snapshot.offsets_len],
        ) {
            *offset = saved_offset;
        }
        for offset in offsets {
            *offset = self.modified.len().min(*offset);
        }
        self.undo_offsets.truncate(snapshot.offsets_start);

        if self.snapshots == 0 {
            let min = self.offsets.iter().copied().min().unwrap_or(0);
            // Any indices still in `modified` may not have been instantiated, so if we observe them again
            // we need to notify any listeners again
            for index in self.modified.drain(..min) {
                self.modified_set.remove(index);
            }
            for offset in &mut self.offsets {
                *offset -= min;
            }
        }
    }

    pub fn commit(&mut self, snapshot: Snapshot<T>) {
        self.snapshots -= 1;
        self.undo_offsets.truncate(snapshot.offsets_start);
        if self.snapshots == 0 {
            // Everything up until this point is committed, so we can forget anything before the
            // current offsets
            let min = self.offsets.iter().copied().min().unwrap_or(0);
            self.modified.drain(..min);
            for offset in &mut self.offsets {
                *offset -= min;
            }
        }
    }

    pub fn register(&mut self) -> Offset<T> {
        let index = self.offsets.len();
        self.offsets.push(0);
        Offset { index, _marker: PhantomData }
    }

    pub fn deregister(&mut self, offset: Offset<T>) {
        assert_eq!(offset.index, self.offsets.len() - 1);
        self.offsets.pop();
        std::mem::forget(offset);
    }
}

#[must_use]
pub struct Offset<T> {
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> Drop for Offset<T> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("Offsets should be deregistered")
        }
    }
}

#[must_use]
#[derive(Debug)]
pub struct Snapshot<T> {
    modified_len: usize,
    offsets_start: usize,
    offsets_len: usize,
    _marker: PhantomData<T>,
}
