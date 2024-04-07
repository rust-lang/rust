//! helper data structure to schedule work for parallel prime caches.
use std::{collections::VecDeque, hash::Hash};

use crate::FxHashMap;

pub(crate) struct TopologicSortIterBuilder<T> {
    nodes: FxHashMap<T, Entry<T>>,
}

impl<T> TopologicSortIterBuilder<T>
where
    T: Copy + Eq + PartialEq + Hash,
{
    fn new() -> Self {
        Self { nodes: Default::default() }
    }

    fn get_or_create_entry(&mut self, item: T) -> &mut Entry<T> {
        self.nodes.entry(item).or_default()
    }

    pub(crate) fn add(&mut self, item: T, predecessors: impl IntoIterator<Item = T>) {
        let mut num_predecessors = 0;

        for predecessor in predecessors.into_iter() {
            self.get_or_create_entry(predecessor).successors.push(item);
            num_predecessors += 1;
        }

        let entry = self.get_or_create_entry(item);
        entry.num_predecessors += num_predecessors;
    }

    pub(crate) fn build(self) -> TopologicalSortIter<T> {
        let ready = self
            .nodes
            .iter()
            .filter_map(
                |(item, entry)| if entry.num_predecessors == 0 { Some(*item) } else { None },
            )
            .collect();

        TopologicalSortIter { nodes: self.nodes, ready }
    }
}

pub(crate) struct TopologicalSortIter<T> {
    ready: VecDeque<T>,
    nodes: FxHashMap<T, Entry<T>>,
}

impl<T> TopologicalSortIter<T>
where
    T: Copy + Eq + PartialEq + Hash,
{
    pub(crate) fn builder() -> TopologicSortIterBuilder<T> {
        TopologicSortIterBuilder::new()
    }

    pub(crate) fn pending(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn mark_done(&mut self, item: T) {
        let entry = self.nodes.remove(&item).expect("invariant: unknown item marked as done");

        for successor in entry.successors {
            let succ_entry = self
                .nodes
                .get_mut(&successor)
                .expect("invariant: unknown successor referenced by entry");

            succ_entry.num_predecessors -= 1;
            if succ_entry.num_predecessors == 0 {
                self.ready.push_back(successor);
            }
        }
    }
}

impl<T> Iterator for TopologicalSortIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.ready.pop_front()
    }
}

struct Entry<T> {
    successors: Vec<T>,
    num_predecessors: usize,
}

impl<T> Default for Entry<T> {
    fn default() -> Self {
        Self { successors: Default::default(), num_predecessors: 0 }
    }
}
