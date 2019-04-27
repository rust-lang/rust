use rustc_data_structures::sync::worker::{Worker, WorkerExecutor};
use rustc_data_structures::sync::{Lrc, AtomicCell};
use rustc_data_structures::{unlikely, cold_path};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_serialize::{Decodable, Encodable, Encoder, Decoder, opaque};
use std::mem;
use std::fs::File;
use std::io::Write;
use std::iter::repeat;
use crate::util::common::time_ext;
use crate::dep_graph::dep_node::{DepKind, DepNode};
use super::prev::PreviousDepGraph;
use super::graph::{DepNodeData, DepNodeIndex, DepNodeState};

pub fn decode_dep_graph(
    time_passes: bool,
    d: &mut opaque::Decoder<'_>,
    results_d: &mut opaque::Decoder<'_>,
) -> Result<(
        PreviousDepGraph,
        IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>,
        Vec<DepNodeIndex>,
    ), String> {
    let fingerprints: IndexVec<DepNodeIndex, Fingerprint> = 
        time_ext(time_passes, None, "decode prev result fingerprints", || {
            IndexVec::decode(results_d)
        })?;
    let mut nodes: IndexVec<_, _> = repeat(DepNode {
            kind: DepKind::Null,
            hash: Fingerprint::ZERO,
    }).take(fingerprints.len()).collect();
    let mut edges: IndexVec<_, _> = repeat(None).take(fingerprints.len()).collect();
    let mut state: IndexVec<_, _> = (0..fingerprints.len()).map(|_| {
            AtomicCell::new(DepNodeState::Invalid)
        }).collect();
    let mut invalid = Vec::new();
    loop {
        if d.position() == d.data.len() {
            break;
        }
        match SerializedAction::decode(d)? {
            SerializedAction::AllocateNodes => {
                let len = d.read_u32()?;
                let start = DepNodeIndex::decode(d)?.as_u32();
                for i in 0..len {
                    let i = DepNodeIndex::from_u32(start + i);
                    let node = DepNode::decode(d)?;
                    nodes[i] = node;
                    edges[i] = Some(Box::<[DepNodeIndex]>::decode(d)?);

                    if likely!(node.kind.is_eval_always()) {
                        state[i] = AtomicCell::new(DepNodeState::Unknown);
                    } else {
                        state[i] = AtomicCell::new(DepNodeState::UnknownEvalAlways);
                    }
                }
            }
            SerializedAction::UpdateEdges => {
                let len = d.read_u32()?;
                for _ in 0..len {
                    let i = DepNodeIndex::decode(d)?;
                    edges[i] = Some(Box::<[DepNodeIndex]>::decode(d)?);
                }
            }
            SerializedAction::InvalidateNodes => {
                let len = d.read_u32()?;
                for _ in 0..len {
                    let i = DepNodeIndex::decode(d)?;
                    state[i] = AtomicCell::new(DepNodeState::Invalid);
                    invalid.push(i);
                }
            }
        }
    }
    let index: FxHashMap<_, _> = time_ext(time_passes, None, "create dep node index", || {
        nodes
            .iter_enumerated()
            .filter(|(idx, _)| *state[*idx].get_mut() != DepNodeState::Invalid)
            .map(|(idx, dep_node)| (*dep_node, idx))
            .collect()
    });
    Ok((PreviousDepGraph {
        index,
        nodes,
        fingerprints,
        edges,
    }, state, invalid))
}

#[derive(Debug, RustcDecodable, RustcEncodable)]
enum SerializedAction {
    AllocateNodes,
    UpdateEdges,
    InvalidateNodes,
}

#[derive(Debug)]
enum Action {
    NewNodes(Vec<(DepNodeIndex, DepNodeData)>),
    UpdateNodes(Vec<(DepNodeIndex, DepNodeData)>),
    // FIXME: Is this redundant since these nodes will be also be updated?
    // Could one of the indirect dependencies of a dep node change its result and
    // cause a red node to be incorrectly green again?
    // What about nodes which are in an unknown state?
    // We must invalidate unknown nodes. Red nodes will have an entry in UpdateNodes
    InvalidateNodes(Vec<DepNodeIndex>),
}

struct SerializerWorker {
    fingerprints: IndexVec<DepNodeIndex, Fingerprint>,
    previous: Lrc<PreviousDepGraph>,
    file: File,
}

impl SerializerWorker {
    fn encode_deps(
        &mut self,
        encoder: &mut opaque::Encoder,
        data: &DepNodeData,
    ) {
        // Encode dependencies
        encoder.emit_u32(data.edges.len() as u32).ok();
        for edge in &data.edges {
            edge.encode(encoder).ok();
        }
    }

    fn encode_new_nodes(
        &mut self,
        encoder: &mut opaque::Encoder,
        nodes: Vec<(DepNodeIndex, DepNodeData)>
    ) {
        // Calculates the number of nodes with indices consecutively increasing by one
        // starting at `i`.
        let run_length = |i: usize| {
            let start = nodes[i].0.as_u32() as usize;
            let mut l = 1;
            loop {
                if i + l >= nodes.len() {
                    return l;
                }
                if nodes[i + l].0.as_u32() as usize != start + l {
                    return l;
                }
                l += 1;
            }
        };

        let mut i = 0;

        loop {
            if i >= nodes.len() {
                break;
            }

            SerializedAction::AllocateNodes.encode(encoder).ok();

            let len = run_length(i);

            // Emit the number of nodes we're emitting
            encoder.emit_u32(len as u32).ok();

            // Emit the dep node index of the first node
            nodes[i].0.encode(encoder).ok();

            for data in &nodes[i..(i+len)] {
                // Update the result fingerprint
                self.set_fingerprint(data.0, data.1.fingerprint);

                // Encode the node
                data.1.node.encode(encoder).ok();

                // Encode dependencies
                self.encode_deps(encoder, &data.1);
            }

            i += len;
        }
    }

    fn encode_updated_nodes(
        &mut self,
        encoder: &mut opaque::Encoder,
        mut nodes: Vec<(DepNodeIndex, DepNodeData)>
    ) {
        // Figure out how many nodes actually changed
        let mut count = 0u32;
        for &mut (ref mut i, ref data) in &mut nodes {
            self.fingerprints[*i] = data.fingerprint;

            if &*data.edges != self.previous.edge_targets_from(*i) {
                count += 1;
            } else {
                // Mark this node as unchanged
                *i = DepNodeIndex::INVALID;
            }
        }
        if count == 0 {
            return;
        }

        SerializedAction::UpdateEdges.encode(encoder).ok();

        encoder.emit_u32(count).ok();

        for (i, data) in nodes {
            if i == DepNodeIndex::INVALID {
                continue;
            }

            // Encode index
            i.encode(encoder).ok();

            // Encode dependencies
            self.encode_deps(encoder, &data);
        }
    }

    fn set_fingerprint(&mut self, i: DepNodeIndex, fingerprint: Fingerprint) {
        let ii = i.as_u32() as usize;
        let len = self.fingerprints.len();
        if ii >= len {
            self.fingerprints.extend(repeat(Fingerprint::ZERO).take(ii - len + 1));
        }
        self.fingerprints[i] = fingerprint;
    }
}

impl Worker for SerializerWorker {
    type Message = (usize, Action);
    type Result = IndexVec<DepNodeIndex, Fingerprint>;

    fn message(&mut self, (buffer_size_est, action): (usize, Action)) {
        let mut encoder = opaque::Encoder::new(Vec::with_capacity(buffer_size_est * 5));
        let action = match action {
            Action::UpdateNodes(nodes) => {
                self.encode_updated_nodes(&mut encoder, nodes)
            }
            Action::NewNodes(nodes) => {
                self.encode_new_nodes(&mut encoder, nodes);
            }
            Action::InvalidateNodes(nodes) => {
                SerializedAction::InvalidateNodes.encode(&mut encoder).ok();
                encoder.emit_u32(nodes.len() as u32).ok();
                for node in nodes {
                    node.encode(&mut encoder).ok();
                }
            },
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
    invalids: Vec<DepNodeIndex>,
    new_buffer: Vec<(DepNodeIndex, DepNodeData)>,
    new_buffer_size: usize,
    updated_buffer: Vec<(DepNodeIndex, DepNodeData)>,
    updated_buffer_size: usize,
}

impl Serializer {
    pub fn new(
        file: File,
        previous: Lrc<PreviousDepGraph>,
        invalids: Vec<DepNodeIndex>,
    ) -> Self {
        Serializer {
            invalids,
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
    fn alloc_index(&mut self) -> DepNodeIndex {
        if let Some(invalid) = self.invalids.pop() {
            // Reuse an invalided index
            invalid
        } else {
            // Create a new index
            let index = self.node_count;
            self.node_count += 1;
            DepNodeIndex::from_u32(index)
        }
    }

    #[inline]
    pub(super) fn serialize_new(&mut self, data: DepNodeData) -> DepNodeIndex {
        let index = self.alloc_index();
        let edges = data.edges.len();
        self.new_buffer.push((index, data));
        self.new_buffer_size += 9 + edges;
        if unlikely!(self.new_buffer_size >= BUFFER_SIZE) {
            cold_path(|| {
                self.flush_new();
            })
        }
        index
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
