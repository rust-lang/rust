use std::marker::PhantomData;

use rustc_index::vec::{Idx, IndexVec};

enum Undo<T> {
    Move { index: T, old: usize },
    Extend { group_index: usize, len: usize },
    NewGroup { index: T },
}

pub struct UnifyLog<T: Idx> {
    unified_vars: IndexVec<T, usize>,
    groups: Vec<Vec<T>>,
    undo_log: Vec<Undo<T>>,
    reference_counts: IndexVec<T, u32>,
    snapshots: usize,
}

fn pick2_mut<T, I: Idx>(self_: &mut [T], a: I, b: I) -> (&mut T, &mut T) {
    let (ai, bi) = (a.index(), b.index());
    assert!(ai != bi);

    if ai < bi {
        let (c1, c2) = self_.split_at_mut(bi);
        (&mut c1[ai], &mut c2[0])
    } else {
        let (c2, c1) = pick2_mut(self_, b, a);
        (c1, c2)
    }
}

impl<T: Idx> UnifyLog<T> {
    pub fn new() -> Self {
        UnifyLog {
            unified_vars: IndexVec::new(),
            groups: Vec::new(),
            undo_log: Vec::new(),
            reference_counts: IndexVec::new(),
            snapshots: 0,
        }
    }

    pub fn unify(&mut self, root: T, other: T) {
        if !self.needs_log(other) {
            return;
        }
        self.unified_vars.ensure_contains_elem(root, usize::max_value);
        self.unified_vars.ensure_contains_elem(other, usize::max_value);
        let mut root_group = self.unified_vars[root];
        let other_group = self.unified_vars[other];

        if other_group == usize::max_value() {
            let root_vec = if root_group == usize::max_value() {
                root_group = self.groups.len();
                self.unified_vars[root] = root_group;
                self.groups.push(Vec::new());
                self.undo_log.push(Undo::NewGroup { index: root });
                self.groups.last_mut().unwrap()
            } else {
                let root_vec = &mut self.groups[root_group];
                self.undo_log.push(Undo::Extend { group_index: root_group, len: root_vec.len() });
                root_vec
            };
            root_vec.push(other);
        } else {
            if root_group == usize::max_value() {
                let group = &mut self.unified_vars[root];
                self.undo_log.push(Undo::Move { index: root, old: *group });
                *group = other_group;
                self.groups[other_group].push(other);
            } else {
                let (root_vec, other_vec) = pick2_mut(&mut self.groups, root_group, other_group);
                self.undo_log.push(Undo::Extend { group_index: root_group, len: root_vec.len() });
                root_vec.extend_from_slice(other_vec);

                if self.reference_counts.get(other).map_or(false, |c| *c != 0) {
                    root_vec.push(other);
                }
            }
        }
    }

    pub fn get(&self, root: T) -> &[T] {
        match self.unified_vars.get(root) {
            Some(group) => match self.groups.get(*group) {
                Some(v) => v,
                None => &[],
            },
            None => &[],
        }
    }

    pub fn needs_log(&self, vid: T) -> bool {
        !self.get(vid).is_empty() || self.reference_counts.get(vid).map_or(false, |c| *c != 0)
    }

    pub fn watch_variable(&mut self, index: T) {
        self.reference_counts.ensure_contains_elem(index, || 0);
        self.reference_counts[index] += 1;
    }

    pub fn unwatch_variable(&mut self, index: T) {
        self.reference_counts[index] -= 1;
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
        for undo in self.undo_log.drain(snapshot.undo_log_len as usize..).rev() {
            match undo {
                Undo::Extend { group_index, len } => {
                    self.groups[group_index].truncate(len as usize)
                }
                Undo::Move { index, old } => self.unified_vars[index] = old,
                Undo::NewGroup { index } => {
                    self.groups.pop();
                    self.unified_vars[index] = usize::max_value();
                }
            }
        }
    }
}

pub struct Snapshot<T: Idx> {
    undo_log_len: u32,
    _marker: PhantomData<T>,
}
