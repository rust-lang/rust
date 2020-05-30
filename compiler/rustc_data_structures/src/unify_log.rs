use rustc_index::vec::{Idx, IndexVec};

use ena::undo_log::{Rollback, UndoLogs};

pub enum Undo<T> {
    Move { index: T, old: usize },
    Extend { group_index: usize, len: usize },
    NewGroup { index: T },
}

/// Tracks which variables (represented by indices) that has been unified with eachother.
/// Since there is often only a few variables that are interesting one must call `watch_variable`
/// before this records any variables unified with that variable.
pub struct UnifyLog<T: Idx> {
    unified_vars: IndexVec<T, usize>,
    groups: Vec<Vec<T>>,
    reference_counts: IndexVec<T, u32>,
}

fn pick2_mut<T, I: Idx>(self_: &mut [T], a: I, b: I) -> (&mut T, &mut T) {
    let (ai, bi) = (a.index(), b.index());
    assert!(ai != bi);

    if ai < bi {
        let (c1, c2) = self_.split_at_mut(bi);
        (&mut c1[ai], &mut c2[0])
    } else {
        let (c1, c2) = self_.split_at_mut(ai);
        (&mut c2[0], &mut c1[bi])
    }
}

impl<T: Idx> UnifyLog<T> {
    pub fn new() -> Self {
        UnifyLog {
            unified_vars: IndexVec::new(),
            groups: Vec::new(),
            reference_counts: IndexVec::new(),
        }
    }

    pub fn unify(&mut self, undo_log: &mut impl UndoLogs<Undo<T>>, root: T, other: T) {
        if !self.needs_log(other) {
            return;
        }
        self.unified_vars.ensure_contains_elem(root.max(other), usize::max_value);
        let mut root_group = self.unified_vars[root];
        let other_group = self.unified_vars[other];

        match (root_group, other_group) {
            (usize::MAX, usize::MAX) => {
                // Neither variable is part of a group, create a new one at the root and associate
                // other
                root_group = self.groups.len();
                self.unified_vars[root] = root_group;
                self.groups.push(vec![other]);
                undo_log.push(Undo::NewGroup { index: root });
            }
            (usize::MAX, _) => {
                // `other` has a group, point `root` to it and associate other
                let group = &mut self.unified_vars[root];
                undo_log.push(Undo::Move { index: root, old: *group });
                *group = other_group;
                self.groups[other_group].push(other);
            }
            (_, usize::MAX) => {
                // `root` hasa group, just associate `other`
                let root_vec = &mut self.groups[root_group];
                undo_log.push(Undo::Extend { group_index: root_group, len: root_vec.len() });
                root_vec.push(other);
            }
            _ => {
                // Both variables has their own groups, associate all of `other` to root
                let (root_vec, other_vec) = pick2_mut(&mut self.groups, root_group, other_group);
                undo_log.push(Undo::Extend { group_index: root_group, len: root_vec.len() });
                root_vec.extend_from_slice(other_vec);

                // We only need to add `other` if there is a watcher for it (there might only be
                // watchers for the other variables in its group)
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
}

impl<I: Idx> Rollback<Undo<I>> for UnifyLog<I> {
    fn reverse(&mut self, undo: Undo<I>) {
        match undo {
            Undo::Extend { group_index, len } => self.groups[group_index].truncate(len as usize),
            Undo::Move { index, old } => self.unified_vars[index] = old,
            Undo::NewGroup { index } => {
                self.groups.pop();
                self.unified_vars[index] = usize::max_value();
            }
        }
    }
}
