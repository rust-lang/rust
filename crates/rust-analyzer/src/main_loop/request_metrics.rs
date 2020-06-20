//! Records stats about requests
use std::time::Duration;

use lsp_server::RequestId;

#[derive(Debug)]
pub(crate) struct RequestMetrics {
    pub(crate) id: RequestId,
    pub(crate) method: String,
    pub(crate) duration: Duration,
}

const N_COMPLETED_REQUESTS: usize = 10;

#[derive(Debug, Default)]
pub(crate) struct LatestRequests {
    // hand-rolling VecDeque here to print things in a nicer way
    buf: [Option<RequestMetrics>; N_COMPLETED_REQUESTS],
    idx: usize,
}

impl LatestRequests {
    pub(crate) fn record(&mut self, request: RequestMetrics) {
        // special case: don't track status request itself
        if request.method == "rust-analyzer/analyzerStatus" {
            return;
        }
        let idx = self.idx;
        self.buf[idx] = Some(request);
        self.idx = (idx + 1) % N_COMPLETED_REQUESTS;
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (bool, &RequestMetrics)> {
        let idx = self.idx;
        self.buf.iter().enumerate().filter_map(move |(i, req)| Some((i == idx, req.as_ref()?)))
    }
}
