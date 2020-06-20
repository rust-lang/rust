//! Manages the set of in-flight requests in both directions.
use std::time::{Duration, Instant};

use lsp_server::RequestId;
use rustc_hash::FxHashMap;
use serde::Serialize;

#[derive(Debug)]
pub(crate) struct ReqQueue<H> {
    pub(crate) incoming: Incoming,
    pub(crate) outgoing: Outgoing<H>,
}

impl<H> Default for ReqQueue<H> {
    fn default() -> Self {
        ReqQueue { incoming: Incoming::default(), outgoing: Outgoing::default() }
    }
}

#[derive(Debug)]
pub(crate) struct Outgoing<H> {
    next: u64,
    pending: FxHashMap<RequestId, H>,
}

impl<H> Default for Outgoing<H> {
    fn default() -> Self {
        Outgoing { next: 0, pending: FxHashMap::default() }
    }
}

impl<H> Outgoing<H> {
    pub(crate) fn register<R>(&mut self, params: R::Params, handler: H) -> lsp_server::Request
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
    {
        let id = RequestId::from(self.next);
        self.next += 1;
        self.pending.insert(id.clone(), handler);
        lsp_server::Request::new(id, R::METHOD.to_string(), params)
    }
    pub(crate) fn complete(&mut self, id: RequestId) -> H {
        self.pending.remove(&id).unwrap()
    }
}

#[derive(Debug)]
pub(crate) struct CompletedInRequest {
    pub(crate) id: RequestId,
    pub(crate) method: String,
    pub(crate) duration: Duration,
}

#[derive(Debug)]
pub(crate) struct PendingInRequest {
    pub(crate) id: RequestId,
    pub(crate) method: String,
    pub(crate) received: Instant,
}

impl From<PendingInRequest> for CompletedInRequest {
    fn from(pending: PendingInRequest) -> CompletedInRequest {
        CompletedInRequest {
            id: pending.id,
            method: pending.method,
            duration: pending.received.elapsed(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct Incoming {
    pending: FxHashMap<RequestId, PendingInRequest>,
}

impl Incoming {
    pub(crate) fn register(&mut self, request: PendingInRequest) {
        let id = request.id.clone();
        let prev = self.pending.insert(id.clone(), request);
        assert!(prev.is_none(), "duplicate request with id {}", id);
    }
    pub(crate) fn cancel(&mut self, id: RequestId) -> Option<lsp_server::Response> {
        if self.pending.remove(&id).is_some() {
            Some(lsp_server::Response::new_err(
                id,
                lsp_server::ErrorCode::RequestCanceled as i32,
                "canceled by client".to_string(),
            ))
        } else {
            None
        }
    }
    pub(crate) fn complete(&mut self, id: RequestId) -> Option<CompletedInRequest> {
        self.pending.remove(&id).map(CompletedInRequest::from)
    }
}

const N_COMPLETED_REQUESTS: usize = 10;

#[derive(Debug, Default)]
pub struct LatestRequests {
    // hand-rolling VecDeque here to print things in a nicer way
    buf: [Option<CompletedInRequest>; N_COMPLETED_REQUESTS],
    idx: usize,
}

impl LatestRequests {
    pub(crate) fn record(&mut self, request: CompletedInRequest) {
        // special case: don't track status request itself
        if request.method == "rust-analyzer/analyzerStatus" {
            return;
        }
        let idx = self.idx;
        self.buf[idx] = Some(request);
        self.idx = (idx + 1) % N_COMPLETED_REQUESTS;
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (bool, &CompletedInRequest)> {
        let idx = self.idx;
        self.buf.iter().enumerate().filter_map(move |(i, req)| Some((i == idx, req.as_ref()?)))
    }
}
