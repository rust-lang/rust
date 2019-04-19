use rustc_data_structures::sync::worker::{Worker, WorkerExecutor};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::{unlikely, cold_path};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_serialize::opaque;
use rustc_serialize::{Decodable, Encodable};
use std::mem;
use std::fs::File;
use std::io::Write;
use super::graph::{DepNodeData, DepNodeIndex, DepNodeState};

newtype_index! {
    pub struct SerializedDepNodeIndex { .. }
}

impl SerializedDepNodeIndex {
    pub fn current(self) -> DepNodeIndex {
        DepNodeIndex::from_u32(self.as_u32())
    }
}

#[derive(Debug, Default)]
pub struct SerializedDepGraph {
    pub(super) nodes: IndexVec<DepNodeIndex, DepNodeData>,
    pub(super) state: IndexVec<DepNodeIndex, DepNodeState>,
}

impl SerializedDepGraph {
    pub fn decode(d: &mut opaque::Decoder<'_>) -> Result<Self, String> {
        let mut nodes = IndexVec::new();
        let mut invalidated_list = Vec::new();
        loop {
            if d.position() == d.data.len() {
                break;
            }
            match Action::decode(d)? {
                Action::NewNodes(new_nodes) => {
                    nodes.extend(new_nodes);
                }
                Action::UpdateNodes(changed) => {
                    for (i, data) in changed {
                        nodes[i] = data;
                    }
                }
                Action::InvalidateNodes(nodes) => {
                    invalidated_list.extend(nodes);
                }
            }
        }
        let mut state: IndexVec<_, _> = (0..nodes.len()).map(|_| {
            DepNodeState::Unknown
        }).collect();
        for i in invalidated_list {
            state[i] = DepNodeState::Invalidated;
        }
        Ok(SerializedDepGraph {
            nodes,
            state,
        })
    }
}

#[derive(Debug, RustcEncodable, RustcDecodable)]
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
    file: File,
}

impl Worker for SerializerWorker {
    type Message = (usize, Action);
    type Result = ();

    fn message(&mut self, (buffer_size_est, action): (usize, Action)) {
        let mut encoder = opaque::Encoder::new(Vec::with_capacity(buffer_size_est * 5));
        action.encode(&mut encoder).ok();
        self.file.write_all(&encoder.into_inner()).expect("unable to write to temp dep graph");
    }

    fn complete(self) {}
}

const BUFFER_SIZE: usize = 800000;

pub struct Serializer {
    worker: Lrc<WorkerExecutor<SerializerWorker>>,
    new_buffer: Vec<DepNodeData>,
    new_buffer_size: usize,
    updated_buffer: Vec<(DepNodeIndex, DepNodeData)>,
    updated_buffer_size: usize,
}

impl Serializer {
    pub fn new(file: File) -> Self {
        Serializer {
            worker: Lrc::new(WorkerExecutor::new(SerializerWorker {
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
    pub(super) fn serialize_new(&mut self, data: DepNodeData) {
        let edges = data.edges.len();
        self.new_buffer.push(data);
        self.new_buffer_size += 8 + edges;
        if unlikely!(self.new_buffer_size >= BUFFER_SIZE) {
            cold_path(|| {
                self.flush_new();
            })
        }
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
    ) {
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
