use rustc_data_structures::sync::worker::{Worker, WorkerExecutor};
use rustc_data_structures::sync::{Lrc, AtomicCell};
use rustc_data_structures::{unlikely, cold_path};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
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

// calcutale a list of bytes to copy from the previous graph

fn decode_bounds(
    d: &mut opaque::Decoder<'_>,
    f: impl FnOnce(&mut opaque::Decoder<'_>) -> Result<(), String>
) -> Result<(usize, usize), String> {
    let start = d.position();
    f(d)?;
    Ok((start, d.position()))
}

type NodePoisitions = Vec<Option<(usize, usize)>>;

fn read_dep_graph_positions(
    d: &mut opaque::Decoder<'_>,
    result: &DecodedDepGraph,
) -> Result<(NodePoisitions, NodePoisitions), String> {
    let node_count = result.prev_graph.nodes.len();
    let mut nodes: NodePoisitions = repeat(None).take(node_count).collect();
    let mut edges: NodePoisitions = repeat(None).take(node_count).collect();

    loop {
        if d.position() == d.data.len() {
            break;
        }
        match SerializedAction::decode(d)? {
            SerializedAction::AllocateNodes => {
                let len = d.read_u32()?;
                let start = DepNodeIndex::decode(d)?.as_u32();
                for i in 0..len {
                    let i = (start + i) as usize;
                    nodes[i] = Some(decode_bounds(d, |d| DepNode::decode(d).map(|_| ()))?);
                    edges[i] = Some(decode_bounds(d, |d| {
                        let len = d.read_u32()?;
                        for _ in 0..len {
                            DepNodeIndex::decode(d)?;
                        }
                        Ok(())
                    })?);
                }
            }
            SerializedAction::UpdateEdges => {
                let len = d.read_u32()?;
                for _ in 0..len {
                    let i = DepNodeIndex::decode(d)?.as_u32() as usize;
                    edges[i] = Some(decode_bounds(d, |d| {
                        let len = d.read_u32()?;
                        for _ in 0..len {
                            DepNodeIndex::decode(d)?;
                        }
                        Ok(())
                    })?);
                }
            }
            SerializedAction::InvalidateNodes => {
                let len = d.read_u32()?;
                for _ in 0..len {
                    let i = DepNodeIndex::decode(d)?.as_u32() as usize;
                    nodes[i] = None;
                    edges[i] = None;
                }
            }
        }
    }

    Ok((nodes, edges))
}

pub fn gc_dep_graph(
    time_passes: bool,
    d: &mut opaque::Decoder<'_>,
    result: &DecodedDepGraph,
    file: &mut File,
) {
    let (nodes, edges) = time_ext(time_passes, None, "read dep-graph positions", || {
        read_dep_graph_positions(d, result).unwrap()
    });

    let mut i = 0;

    loop {
        // Skip empty nodes
        while i < nodes.len() && nodes[i].is_none() {
            i += 1;
        }

        // Break if we are done
        if i >= nodes.len() {
            break;
        }

        // Find out how many consecutive nodes we will emit
        let mut len = 1;
        while i + len < nodes.len() && nodes[i + len].is_some() {
            len += 1;
        }

        let mut encoder = opaque::Encoder::new(Vec::with_capacity(11));
        SerializedAction::AllocateNodes.encode(&mut encoder).ok();
        // Emit the number of nodes we're emitting
        encoder.emit_u32(len as u32).ok();

        // Emit the dep node index of the first node
        DepNodeIndex::new(i).encode(&mut encoder).ok();

        file.write_all(&encoder.into_inner()).unwrap();

        let mut buffers = Vec::with_capacity(nodes.len() * 2);

        let push_buffer = |buffers: &mut Vec<(usize, usize)>, range: (usize, usize)| {
            if let Some(last) = buffers.last_mut() {
                if last.1 == range.0 {
                    // Extend the last range
                    last.1 = range.1;
                    return;
                }
            }
            buffers.push(range);
        };

        for i in i..(i + len) {
            // Encode the node
            push_buffer(&mut buffers, nodes[i].unwrap());

            // Encode dependencies
            push_buffer(&mut buffers, edges[i].unwrap());
        }

        for buffer in buffers {
            file.write_all(&d.data[buffer.0..buffer.1]).unwrap();
        }

        i += len;
    }
}

/// A simpler dep graph used for debugging and testing purposes.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable, Default)]
pub struct DepGraphModel {
    pub data: FxHashMap<DepNodeIndex, DepNodeData>,
}

impl DepGraphModel {
    fn apply(&mut self, action: &Action) {
        match action {
            Action::UpdateNodes(nodes) => {
                for n in nodes {
                    self.data
                        .entry(n.0)
                        .or_insert_with(|| panic!()).edges = n.1.edges.clone();
                }
            }
            Action::NewNodes(nodes) => {
                for n in nodes {
                    assert!(self.data.insert(n.0, n.1.clone()).is_none());
                }
            }
            Action::InvalidateNodes(nodes) => {
                for n in nodes {
                    assert!(self.data.remove(&n).is_some());
                }
            },
        }
    }
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable, Default)]
pub struct CompletedDepGraph {
    /// Hashes of the results of dep nodes
    pub(super) results: IndexVec<DepNodeIndex, Fingerprint>,
    /// A simpler dep graph stored alongside the result for debugging purposes.
    /// This is also constructed when we want to query the dep graph.
    pub model: Option<DepGraphModel>,
}

pub struct DecodedDepGraph {
    pub prev_graph: PreviousDepGraph,
    pub state: IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>,
    pub invalidated: Vec<DepNodeIndex>,
    pub needs_gc: bool,
    pub model: Option<DepGraphModel>,
}

impl DecodedDepGraph {
    /// Asserts that the model matches the real dep graph we decoded
    fn validate_model(&mut self) {
        let model = if let Some(ref model) = self.model {
            model
        } else {
            return
        };

        for i in self.state.indices() {
            if *self.state[i].get_mut() == DepNodeState::Invalid {
                assert!(!model.data.contains_key(&i));
                assert_eq!(self.prev_graph.edges[i], None);
            } else {
                let data = model.data.get(&i).unwrap();
                assert_eq!(self.prev_graph.nodes[i], data.node);
                assert_eq!(&self.prev_graph.edges[i].as_ref().unwrap()[..], &data.edges[..]);
            }
        }

        for k in model.data.keys() {
            assert!((k.as_u32() as usize) < self.state.len());
        }

    }
}

pub fn decode_dep_graph(
    time_passes: bool,
    d: &mut opaque::Decoder<'_>,
    results_d: &mut opaque::Decoder<'_>,
) -> Result<DecodedDepGraph, String> {
    // Metrics used to decide when to GC
    let mut valid_data = 0;
    let mut total_data = 0;

    let result_format = time_ext(time_passes, None, "decode prev result fingerprints", || {
        CompletedDepGraph::decode(results_d)
    })?;

    let node_count = result_format.results.len();
    let mut nodes: IndexVec<_, _> = repeat(DepNode {
            kind: DepKind::Null,
            hash: Fingerprint::ZERO,
    }).take(node_count).collect();
    let mut edges: IndexVec<_, _> = repeat(None).take(node_count).collect();
    let mut state: IndexVec<_, _> = (0..node_count).map(|_| {
            AtomicCell::new(DepNodeState::Invalid)
        }).collect();
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
                    let node_edges = Box::<[DepNodeIndex]>::decode(d)?;
                    valid_data += node_edges.len();
                    total_data += node_edges.len();
                    edges[i] = Some(node_edges);

                    if unlikely!(node.kind.is_eval_always()) {
                        state[i] = AtomicCell::new(DepNodeState::UnknownEvalAlways);
                    } else {
                        state[i] = AtomicCell::new(DepNodeState::Unknown);
                    }
                }
                valid_data += len as usize * 8;
                total_data += len as usize * 8;
            }
            SerializedAction::UpdateEdges => {
                let len = d.read_u32()?;
                for _ in 0..len {
                    let i = DepNodeIndex::decode(d)?;
                    valid_data -= edges[i].as_ref().map_or(0, |edges| edges.len());
                    let node_edges = Box::<[DepNodeIndex]>::decode(d)?;
                    valid_data += node_edges.len();
                    total_data += node_edges.len();
                    edges[i] = Some(node_edges);
                }
            }
            SerializedAction::InvalidateNodes => {
                let len = d.read_u32()?;
                for _ in 0..len {
                    let i = DepNodeIndex::decode(d)?;
                    valid_data -= edges[i].as_ref().map_or(0, |edges| edges.len());
                    state[i] = AtomicCell::new(DepNodeState::Invalid);
                    edges[i] = None;
                }
                valid_data -= len as usize * 8;
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

    debug!(
        "valid bytes {} total bytes {} ratio {}",
        valid_data,
        total_data,
        valid_data as f32 / total_data as f32
    );

    let mut graph = DecodedDepGraph {
        prev_graph: PreviousDepGraph {
            index,
            nodes,
            fingerprints: result_format.results,
            edges,
        },
        invalidated: state.indices()
            .filter(|&i| *state[i].get_mut() == DepNodeState::Invalid)
            .collect(),
        state,
        needs_gc: valid_data + valid_data / 3 < total_data,
        model: result_format.model,
    };

    graph.validate_model();

    Ok(graph)
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
    file: Option<File>,
    model: Option<DepGraphModel>,
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
    type Result = CompletedDepGraph;

    fn message(&mut self, (buffer_size_est, action): (usize, Action)) {
        // Apply the action to the model if present
        self.model.as_mut().map(|model| model.apply(&action));

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
        self.file.as_mut().map(|file| {
            file.write_all(&encoder.into_inner()).expect("unable to write to temp dep graph");
        });
    }

    fn complete(self) -> CompletedDepGraph {
        CompletedDepGraph {
            results: self.fingerprints,
            model: self.model
        }
    }
}

const BUFFER_SIZE: usize = 800000;

pub struct Serializer {
    worker: Lrc<WorkerExecutor<SerializerWorker>>,
    node_count: u32,
    invalidated: Vec<DepNodeIndex>,
    new_buffer: Vec<(DepNodeIndex, DepNodeData)>,
    new_buffer_size: usize,
    updated_buffer: Vec<(DepNodeIndex, DepNodeData)>,
    updated_buffer_size: usize,
}

impl Serializer {
    pub fn new(
        file: Option<File>,
        previous: Lrc<PreviousDepGraph>,
        invalidated: Vec<DepNodeIndex>,
        model: Option<DepGraphModel>,
    ) -> Self {
        Serializer {
            invalidated,
            node_count: previous.nodes.len() as u32,
            worker: Lrc::new(WorkerExecutor::new(SerializerWorker {
                fingerprints: previous.fingerprints.clone(),
                previous,
                file,
                model,
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
        if let Some(invalidated) = self.invalidated.pop() {
            // Reuse an invalided index
            invalidated
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

    pub(super) fn complete(
        &mut self,
        invalidate: Vec<DepNodeIndex>,
    ) -> CompletedDepGraph {
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
