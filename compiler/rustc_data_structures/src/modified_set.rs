use std::{collections::VecDeque, marker::PhantomData};

use rustc_index::{bit_set::BitSet, vec::Idx};

#[derive(Copy, Clone, Debug)]
enum Undo<T> {
    Add(T),
    Drain { index: usize, offset: usize },
}

#[derive(Clone, Debug)]
pub struct ModifiedSet<T: Idx> {
    modified: VecDeque<Undo<T>>,
    snapshots: usize,
    modified_set: BitSet<T>,
    offsets: Vec<usize>,
}

impl<T: Idx> Default for ModifiedSet<T> {
    fn default() -> Self {
        Self {
            modified: Default::default(),
            snapshots: 0,
            modified_set: BitSet::new_empty(0),
            offsets: Vec::new(),
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
            self.modified.push_back(Undo::Add(index));
        }
    }

    pub fn drain(&mut self, index: &Offset<T>, mut f: impl FnMut(T) -> bool) {
        let offset = &mut self.offsets[index.index];
        if *offset < self.modified.len() {
            for &undo in self.modified.iter().skip(*offset) {
                if let Undo::Add(index) = undo {
                    f(index);
                }
            }
            self.modified.push_back(Undo::Drain { index: index.index, offset: *offset });
            *offset = self.modified.len();
        }
    }

    pub fn snapshot(&mut self) -> Snapshot<T> {
        self.snapshots += 1;
        Snapshot { modified_len: self.modified.len(), _marker: PhantomData }
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot<T>) {
        self.snapshots -= 1;
        if snapshot.modified_len < self.modified.len() {
            for &undo in
                self.modified.iter().rev().take(self.modified.len() - snapshot.modified_len)
            {
                match undo {
                    Undo::Add(index) => {
                        self.modified_set.remove(index);
                    }
                    Undo::Drain { index, offset } => {
                        if let Some(o) = self.offsets.get_mut(index) {
                            *o = offset;
                        }
                    }
                }
            }
            self.modified.truncate(snapshot.modified_len);
        }

        if self.snapshots == 0 {
            let min = self.offsets.iter().copied().min().unwrap_or(0);
            self.modified.drain(..min);
            for offset in &mut self.offsets {
                *offset -= min;
            }
        }
    }

    pub fn commit(&mut self, _snapshot: Snapshot<T>) {
        self.snapshots -= 1;
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
        self.modified.push_back(Undo::Drain { index, offset: 0 });
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
    _marker: PhantomData<T>,
}
