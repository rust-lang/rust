// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ich::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::StableHasher;
use std::env;
use std::hash::Hash;
use std::mem;
use super::{DepGraphQuery, DepKind, DepNode};
use super::debug::EdgeFilter;

pub struct DepGraphEdges {
    nodes: Vec<DepNode>,
    indices: FxHashMap<DepNode, IdIndex>,
    edges: FxHashSet<(IdIndex, IdIndex)>,
    task_stack: Vec<OpenTask>,
    forbidden_edge: Option<EdgeFilter>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct IdIndex {
    index: u32
}

impl IdIndex {
    fn new(v: usize) -> IdIndex {
        assert!((v & 0xFFFF_FFFF) == v);
        IdIndex { index: v as u32 }
    }

    fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
enum OpenTask {
    Regular {
        node: DepNode,
        reads: Vec<DepNode>,
        read_set: FxHashSet<DepNode>,
    },
    Anon {
        reads: Vec<DepNode>,
        read_set: FxHashSet<DepNode>,
    },
    Ignore,
}

impl DepGraphEdges {
    pub fn new() -> DepGraphEdges {
        let forbidden_edge = if cfg!(debug_assertions) {
            match env::var("RUST_FORBID_DEP_GRAPH_EDGE") {
                Ok(s) => {
                    match EdgeFilter::new(&s) {
                        Ok(f) => Some(f),
                        Err(err) => bug!("RUST_FORBID_DEP_GRAPH_EDGE invalid: {}", err),
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        DepGraphEdges {
            nodes: vec![],
            indices: FxHashMap(),
            edges: FxHashSet(),
            task_stack: Vec::new(),
            forbidden_edge,
        }
    }

    fn id(&self, index: IdIndex) -> DepNode {
        self.nodes[index.index()]
    }

    pub fn push_ignore(&mut self) {
        self.task_stack.push(OpenTask::Ignore);
    }

    pub fn pop_ignore(&mut self) {
        let popped_node = self.task_stack.pop().unwrap();
        debug_assert_eq!(popped_node, OpenTask::Ignore);
    }

    pub fn push_task(&mut self, key: DepNode) {
        self.task_stack.push(OpenTask::Regular {
            node: key,
            reads: Vec::new(),
            read_set: FxHashSet(),
        });
    }

    pub fn pop_task(&mut self, key: DepNode) {
        let popped_node = self.task_stack.pop().unwrap();

        if let OpenTask::Regular {
            node,
            read_set: _,
            reads
        } = popped_node {
            debug_assert_eq!(node, key);

            let target_id = self.get_or_create_node(node);

            for read in reads.into_iter() {
                let source_id = self.get_or_create_node(read);
                self.edges.insert((source_id, target_id));
            }
        } else {
            bug!("pop_task() - Expected regular task to be popped")
        }
    }

    pub fn push_anon_task(&mut self) {
        self.task_stack.push(OpenTask::Anon {
            reads: Vec::new(),
            read_set: FxHashSet(),
        });
    }

    pub fn pop_anon_task(&mut self, kind: DepKind) -> DepNode {
        let popped_node = self.task_stack.pop().unwrap();

        if let OpenTask::Anon {
            read_set: _,
            reads
        } = popped_node {
            let mut fingerprint = Fingerprint::zero();
            let mut hasher = StableHasher::new();

            for read in reads.iter() {
                mem::discriminant(&read.kind).hash(&mut hasher);

                // Fingerprint::combine() is faster than sending Fingerprint
                // through the StableHasher (at least as long as StableHasher
                // is so slow).
                fingerprint = fingerprint.combine(read.hash);
            }

            fingerprint = fingerprint.combine(hasher.finish());

            let target_dep_node = DepNode {
                kind,
                hash: fingerprint,
            };

            if self.indices.contains_key(&target_dep_node) {
                return target_dep_node;
            }

            let target_id = self.get_or_create_node(target_dep_node);

            for read in reads.into_iter() {
                let source_id = self.get_or_create_node(read);
                self.edges.insert((source_id, target_id));
            }

            target_dep_node
        } else {
            bug!("pop_anon_task() - Expected anonymous task to be popped")
        }
    }

    /// Indicates that the current task `C` reads `v` by adding an
    /// edge from `v` to `C`. If there is no current task, has no
    /// effect. Note that *reading* from tracked state is harmless if
    /// you are not in a task; what is bad is *writing* to tracked
    /// state (and leaking data that you read into a tracked task).
    pub fn read(&mut self, source: DepNode) {
        match self.task_stack.last_mut() {
            Some(&mut OpenTask::Regular {
                node: target,
                ref mut reads,
                ref mut read_set,
            }) => {
                if read_set.insert(source) {
                    reads.push(source);

                    if cfg!(debug_assertions) {
                        if let Some(ref forbidden_edge) = self.forbidden_edge {
                            if forbidden_edge.test(&source, &target) {
                                bug!("forbidden edge {:?} -> {:?} created", source, target)
                            }
                        }
                    }
                }
            }
            Some(&mut OpenTask::Anon {
                ref mut reads,
                ref mut read_set,
            }) => {
                if read_set.insert(source) {
                    reads.push(source);
                }
            }
            Some(&mut OpenTask::Ignore) | None => {
                // ignore
            }
        }
    }

    pub fn query(&self) -> DepGraphQuery {
        let edges: Vec<_> = self.edges.iter()
                                      .map(|&(i, j)| (self.id(i), self.id(j)))
                                      .collect();
        DepGraphQuery::new(&self.nodes, &edges)
    }

    #[inline]
    pub fn add_edge(&mut self, source: DepNode, target: DepNode) {
        let source = self.get_or_create_node(source);
        let target = self.get_or_create_node(target);
        self.edges.insert((source, target));
    }

    pub fn add_node(&mut self, node: DepNode) {
        self.get_or_create_node(node);
    }

    #[inline]
    fn get_or_create_node(&mut self, dep_node: DepNode) -> IdIndex {
        let DepGraphEdges {
            ref mut indices,
            ref mut nodes,
            ..
        } = *self;

        *indices.entry(dep_node).or_insert_with(|| {
            let next_id = nodes.len();
            nodes.push(dep_node);
            IdIndex::new(next_id)
        })
     }
}
