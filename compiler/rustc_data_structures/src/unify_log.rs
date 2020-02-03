use std::marker::PhantomData;

use rustc_index::vec::{Idx, IndexVec};

pub struct UnifyLog<T: Idx> {
    unified_vars: IndexVec<T, Vec<T>>,
    undo_log: Vec<(T, u32)>,
    snapshots: usize,
}

impl<T: Idx> UnifyLog<T> {
    pub fn new() -> Self {
        UnifyLog { unified_vars: IndexVec::new(), undo_log: Vec::new(), snapshots: 0 }
    }

    pub fn unify(&mut self, root: T, other: T) {
        self.unified_vars.ensure_contains_elem(root, Vec::new);
        self.unified_vars.ensure_contains_elem(other, Vec::new);
        let (root_ids, other_ids) = self.unified_vars.pick2_mut(root, other);
        self.undo_log.push((root, root_ids.len() as u32));
        for &other in &*other_ids {
            if !root_ids.contains(&other) {
                root_ids.push(other);
            }
        }
        root_ids.push(other);
    }

    pub fn get(&self, root: T) -> &[T] {
        self.unified_vars.get(root).map(|v| &v[..]).unwrap_or(&[][..])
    }

    pub fn snapshot(&mut self) -> Snapshot<T> {
        self.snapshots += 1;
        Snapshot { undo_log_len: self.undo_log.len() as u32, _marker: PhantomData }
    }

    pub fn commit(&mut self, _snapshot: Snapshot<T>) {
        self.snapshots -= 1;
        if self.snapshots == 0 {
            self.undo_log.clear();
        }
    }

    pub fn rollback_to(&mut self, snapshot: Snapshot<T>) {
        self.snapshots -= 1;
        for (index, len) in self.undo_log.drain(snapshot.undo_log_len as usize..) {
            self.unified_vars[index].truncate(len as usize);
        }
    }
}

pub struct Snapshot<T: Idx> {
    undo_log_len: u32,
    _marker: PhantomData<T>,
}
