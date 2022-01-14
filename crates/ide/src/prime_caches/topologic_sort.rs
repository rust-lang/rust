use std::{collections::VecDeque, hash::Hash};

use rustc_hash::FxHashMap;

pub struct TopologicSortIterBuilder<T> {
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

    pub fn add(&mut self, item: T, predecessors: impl IntoIterator<Item = T>) {
        let mut num_predecessors = 0;

        for predecessor in predecessors.into_iter() {
            self.get_or_create_entry(predecessor).successors.push(item);
            num_predecessors += 1;
        }

        let entry = self.get_or_create_entry(item);
        entry.num_predecessors += num_predecessors;
    }

    pub fn build(self) -> TopologicalSortIter<T> {
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

pub struct TopologicalSortIter<T> {
    ready: VecDeque<T>,
    nodes: FxHashMap<T, Entry<T>>,
}

impl<T> TopologicalSortIter<T>
where
    T: Copy + Eq + PartialEq + Hash,
{
    pub fn builder() -> TopologicSortIterBuilder<T> {
        TopologicSortIterBuilder::new()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn mark_done(&mut self, item: T) {
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
