use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;

#[derive(Debug)]
pub struct CompletedRequest {
    pub id: u64,
    pub method: String,
    pub duration: Duration,
}

#[derive(Debug)]
pub(crate) struct PendingRequest {
    pub(crate) id: u64,
    pub(crate) method: String,
    pub(crate) received: Instant,
}

impl From<PendingRequest> for CompletedRequest {
    fn from(pending: PendingRequest) -> CompletedRequest {
        CompletedRequest {
            id: pending.id,
            method: pending.method,
            duration: pending.received.elapsed(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PendingRequests {
    map: FxHashMap<u64, PendingRequest>,
}

impl PendingRequests {
    pub(crate) fn start(&mut self, request: PendingRequest) {
        let id = request.id;
        let prev = self.map.insert(id, request);
        assert!(prev.is_none(), "duplicate request with id {}", id);
    }
    pub(crate) fn cancel(&mut self, id: u64) -> bool {
        self.map.remove(&id).is_some()
    }
    pub(crate) fn finish(&mut self, id: u64) -> Option<CompletedRequest> {
        self.map.remove(&id).map(CompletedRequest::from)
    }
}

const N_COMPLETED_REQUESTS: usize = 10;

#[derive(Debug, Default)]
pub struct LatestRequests {
    // hand-rolling VecDeque here to print things in a nicer way
    buf: [Option<CompletedRequest>; N_COMPLETED_REQUESTS],
    idx: usize,
}

impl LatestRequests {
    pub(crate) fn record(&mut self, request: CompletedRequest) {
        // special case: don't track status request itself
        if request.method == "rust-analyzer/analyzerStatus" {
            return;
        }
        let idx = self.idx;
        self.buf[idx] = Some(request);
        self.idx = (idx + 1) % N_COMPLETED_REQUESTS;
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (bool, &CompletedRequest)> {
        let idx = self.idx;
        self.buf.iter().enumerate().filter_map(move |(i, req)| Some((i == idx, req.as_ref()?)))
    }
}
