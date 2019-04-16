use rustc_data_structures::sync::worker::{Worker, WorkerExecutor};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::{unlikely, cold_path};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_serialize::opaque;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::mem;
use std::fs::File;
use std::io::Write;
use super::graph::DepNodeData;
use crate::dep_graph::DepNode;
use crate::ich::Fingerprint;

newtype_index! {
    pub struct SerializedDepNodeIndex { .. }
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug, Default)]
pub struct SerializedDepGraph {
    pub nodes: IndexVec<SerializedDepNodeIndex, SerializedNode>,
}

impl Decodable for SerializedDepGraph {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        let mut nodes = IndexVec::new();
        loop {
            let count = d.read_usize()?;
            if count == 0 {
                break;
            }
            for _ in 0..count {
                nodes.push(SerializedNode::decode(d)?);
            }
        }
        Ok(SerializedDepGraph {
            nodes,
        })
    }
}

#[derive(Debug, RustcDecodable)]
pub struct SerializedNode {
    pub node: DepNode,
    pub deps: Vec<SerializedDepNodeIndex>,
    pub fingerprint: Fingerprint,
}

struct SerializerWorker {
    file: File,
}

impl Worker for SerializerWorker {
    type Message = (usize, Vec<DepNodeData>);
    type Result = ();

    fn message(&mut self, (buffer_size_est, nodes): (usize, Vec<DepNodeData>)) {
        let mut encoder = opaque::Encoder::new(Vec::with_capacity(buffer_size_est * 4));
        assert!(!nodes.is_empty());
        encoder.emit_usize(nodes.len()).ok();
        for data in nodes {
            data.node.encode(&mut encoder).ok();
            data.edges.encode(&mut encoder).ok();
            data.fingerprint.encode(&mut encoder).ok();
        }
        self.file.write_all(&encoder.into_inner()).expect("unable to write to temp dep graph");
    }

    fn complete(mut self) {
        let mut encoder = opaque::Encoder::new(Vec::with_capacity(16));
        encoder.emit_usize(0).ok();
        self.file.write_all(&encoder.into_inner()).expect("unable to write to temp dep graph");
    }
}

const BUFFER_SIZE: usize = 800000;

pub struct Serializer {
    worker: Lrc<WorkerExecutor<SerializerWorker>>,
    buffer: Vec<DepNodeData>,
    buffer_size: usize,
}

impl Serializer {
    pub fn new(file: File) -> Self {
        Serializer {
            worker: Lrc::new(WorkerExecutor::new(SerializerWorker {
                file,
            })),
            buffer: Vec::with_capacity(BUFFER_SIZE),
            buffer_size: 0,
        }
    }

    fn flush(&mut self) {
        let msgs = mem::replace(&mut self.buffer, Vec::with_capacity(BUFFER_SIZE));
        let buffer_size = self.buffer_size;
        self.buffer_size = 0;
        self.worker.message_in_pool((buffer_size, msgs));
    }

    #[inline]
    pub(super) fn serialize(&mut self, data: DepNodeData) {
        let edges = data.edges.len();
        self.buffer.push(data);
        self.buffer_size += 8 + edges;
        if unlikely!(self.buffer_size >= BUFFER_SIZE) {
            cold_path(|| {
                self.flush();
            })
        }
    }

    pub fn complete(&mut self) {
        if self.buffer.len() > 0 {
            self.flush();
        }
        self.worker.complete()
    }
}
