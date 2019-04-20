use rustc_data_structures::sync::worker::{Worker, WorkerExecutor};
use rustc_data_structures::sync::{Lrc, AtomicCell};
use rustc_data_structures::{unlikely, cold_path};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_serialize::{Decodable, Encodable, opaque};
use std::mem;
use std::fs::File;
use std::io::Write;
use crate::dep_graph::dep_node::DepNode;
use super::prev::PreviousDepGraph;
use super::graph::{DepNodeData, DepNodeIndex, DepNodeState};

#[derive(Debug, Default)]
pub struct SerializedDepGraph {
    pub(super) nodes: IndexVec<DepNodeIndex, SerializedDepNodeData>,
    pub(super) fingerprints: IndexVec<DepNodeIndex, Fingerprint>,
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub(super) struct SerializedDepNodeData {
    pub(super) node: DepNode,
    pub(super) edges: Vec<DepNodeIndex>,
}

impl SerializedDepGraph {
    pub fn decode(
        d: &mut opaque::Decoder<'_>,
        results_d: &mut opaque::Decoder<'_>,
    ) -> Result<(Self, IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>), String> {
        let fingerprints: IndexVec<DepNodeIndex, Fingerprint> = IndexVec::decode(results_d)?;
        let mut nodes = IndexVec::with_capacity(fingerprints.len());
        let mut state: IndexVec<_, _> = (0..fingerprints.len()).map(|_| {
            AtomicCell::new(DepNodeState::Unknown)
        }).collect();
        loop {
            if d.position() == d.data.len() {
                break;
            }
            match SerializedAction::decode(d)? {
                SerializedAction::NewNodes(new_nodes) => {
                    for (i, data) in new_nodes.iter().enumerate() {
                        // Mark the result of eval_always nodes as invalid so they will
                        // get executed again.
                        if unlikely!(data.node.kind.is_eval_always()) {
                            let idx = DepNodeIndex::new(nodes.len() + i);
                            state[idx] = AtomicCell::new(DepNodeState::Invalid);
                        }
                    }
                    nodes.extend(new_nodes);
                }
                SerializedAction::UpdateEdges(changed) => {
                    for (i, edges) in changed {
                        // Updated results are valid again, except for eval_always nodes
                        // which always start out invalid.
                        if likely!(!nodes[i].node.kind.is_eval_always()) {
                            state[i] = AtomicCell::new(DepNodeState::Unknown);
                        }
                        nodes[i].edges = edges;
                    }
                }
                SerializedAction::InvalidateNodes(nodes) => {
                    for i in nodes {
                        state[i] = AtomicCell::new(DepNodeState::Invalid);
                    }
                }
            }
        }
        Ok((SerializedDepGraph {
            nodes,
            fingerprints,
        }, state))
    }
}

#[derive(Debug, RustcDecodable, RustcEncodable)]
enum SerializedAction {
    NewNodes(Vec<SerializedDepNodeData>),
    UpdateEdges(Vec<(DepNodeIndex, Vec<DepNodeIndex>)>),
    InvalidateNodes(Vec<DepNodeIndex>)
}

#[derive(Debug)]
enum Action {
    NewNodes(Vec<DepNodeData>),
    UpdateNodes(Vec<(DepNodeIndex, DepNodeData)>),
    // FIXME: Is this redundant since these nodes will be also be updated?
    // Could one of the indirect dependencies of a dep node change its result and
    // cause a red node to be incorrectly green again?
    // What about nodes which are in an unknown state?
    // We must invalidate unknown nodes. Red nodes will have an entry in UpdateNodes
    InvalidateNodes(Vec<DepNodeIndex>)
}

struct SerializerWorker {
    fingerprints: IndexVec<DepNodeIndex, Fingerprint>,
    previous: Lrc<PreviousDepGraph>,
    file: File,
}

impl Worker for SerializerWorker {
    type Message = (usize, Action);
    type Result = IndexVec<DepNodeIndex, Fingerprint>;

    fn message(&mut self, (buffer_size_est, action): (usize, Action)) {
        let mut encoder = opaque::Encoder::new(Vec::with_capacity(buffer_size_est * 5));
        let action = match action {
            Action::UpdateNodes(nodes) => {
                SerializedAction::UpdateEdges(nodes.into_iter().filter(|&(i, ref data)| {
                    self.fingerprints[i] = data.fingerprint;
                    // Only emit nodes which actually changed
                    &*data.edges != self.previous.edge_targets_from(i)
                }).map(|(i, data)| (i, data.edges.into_iter().collect::<Vec<_>>())).collect())
            }
            Action::NewNodes(nodes) => {
                SerializedAction::NewNodes(nodes.into_iter().map(|data| {
                    self.fingerprints.push(data.fingerprint);
                    SerializedDepNodeData {
                        node: data.node,
                        edges: data.edges.into_iter().collect(),
                    }
                }).collect())
            }
            Action::InvalidateNodes(nodes) => SerializedAction::InvalidateNodes(nodes),
        };
        action.encode(&mut encoder).ok();
        self.file.write_all(&encoder.into_inner()).expect("unable to write to temp dep graph");
    }

    fn complete(self) -> IndexVec<DepNodeIndex, Fingerprint> {
        self.fingerprints
    }
}

const BUFFER_SIZE: usize = 800000;

pub struct Serializer {
    worker: Lrc<WorkerExecutor<SerializerWorker>>,
    node_count: u32,
    new_buffer: Vec<DepNodeData>,
    new_buffer_size: usize,
    updated_buffer: Vec<(DepNodeIndex, DepNodeData)>,
    updated_buffer_size: usize,
}

impl Serializer {
    pub fn new(
        file: File,
        previous: Lrc<PreviousDepGraph>,
    ) -> Self {
        Serializer {
            node_count: previous.node_count() as u32,
            worker: Lrc::new(WorkerExecutor::new(SerializerWorker {
                fingerprints: previous.fingerprints.clone(),
                previous,
                file,
            })),
            new_buffer: Vec::with_capacity(BUFFER_SIZE),
            new_buffer_size: 0,
            updated_buffer: Vec::with_capacity(BUFFER_SIZE),
            updated_buffer_size: 0,
        }
    }

    fn flush_new(&mut self) {
        let msgs = mem::replace(&mut self.new_buffer, Vec::with_capacity(BUFFER_SIZE));
        let buffer_size = self.new_buffer_size;
        self.new_buffer_size = 0;
        self.worker.message_in_pool((buffer_size, Action::NewNodes(msgs)));
    }

    #[inline]
    pub(super) fn serialize_new(&mut self, data: DepNodeData) -> DepNodeIndex {
        let edges = data.edges.len();
        self.new_buffer.push(data);
        self.new_buffer_size += 8 + edges;
        if unlikely!(self.new_buffer_size >= BUFFER_SIZE) {
            cold_path(|| {
                self.flush_new();
            })
        }
        let index = self.node_count;
        self.node_count += 1;
        DepNodeIndex::from_u32(index)
    }

    fn flush_updated(&mut self) {
        let msgs = mem::replace(&mut self.updated_buffer, Vec::with_capacity(BUFFER_SIZE));
        let buffer_size = self.updated_buffer_size;
        self.updated_buffer_size = 0;
        self.worker.message_in_pool((buffer_size, Action::UpdateNodes(msgs)));
    }

    #[inline]
    pub(super) fn serialize_updated(&mut self, index: DepNodeIndex, data: DepNodeData) {
        let edges = data.edges.len();
        self.updated_buffer.push((index, data));
        self.updated_buffer_size += 9 + edges;
        if unlikely!(self.updated_buffer_size >= BUFFER_SIZE) {
            cold_path(|| {
                self.flush_updated();
            })
        }
    }

    pub fn complete(
        &mut self,
        invalidate: Vec<DepNodeIndex>,
    ) -> IndexVec<DepNodeIndex, Fingerprint> {
        if self.new_buffer.len() > 0 {
            self.flush_new();
        }
        if self.updated_buffer.len() > 0 {
            self.flush_updated();
        }
        if !invalidate.is_empty() {
            self.worker.message_in_pool(
                (invalidate.len(), Action::InvalidateNodes(invalidate))
            );
        }
        self.worker.complete()
    }
}
