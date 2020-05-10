use super::lsp_utils::{notification_new, request_new};
use crossbeam_channel::Sender;
use lsp_server::{Message, RequestId};
use lsp_types::{
    WorkDoneProgress, WorkDoneProgressBegin, WorkDoneProgressCreateParams, WorkDoneProgressEnd,
    WorkDoneProgressReport,
};

const PRIME_CACHES_PROGRESS_TOKEN: &str = "rustAnalyzer/primeCaches";
const WORKSPACE_ANALYSIS_PROGRESS_TOKEN: &str = "rustAnalyzer/workspaceAnalysis";

#[derive(Debug)]
pub(crate) struct PrimeCachesProgressNotifier(ProgressNotifier);

impl Drop for PrimeCachesProgressNotifier {
    fn drop(&mut self) {
        self.0.end("done priming caches".to_owned());
    }
}

impl PrimeCachesProgressNotifier {
    pub(crate) fn begin(sender: Sender<Message>, req_id: RequestId, total: usize) -> Self {
        let me = Self(ProgressNotifier {
            sender,
            processed: 0,
            total,
            token: PRIME_CACHES_PROGRESS_TOKEN,
            label: "priming caches",
        });
        me.0.begin(req_id);
        me
    }

    pub(crate) fn report(&mut self, processed: usize) -> IsDone {
        self.0.report(processed)
    }
}

#[derive(Debug)]
pub(crate) struct WorkspaceAnalysisProgressNotifier(ProgressNotifier);

impl Drop for WorkspaceAnalysisProgressNotifier {
    fn drop(&mut self) {
        self.0.end("done analyzing workspace".to_owned());
    }
}

impl WorkspaceAnalysisProgressNotifier {
    pub(crate) fn begin(sender: Sender<Message>, req_id: RequestId, total: usize) -> Self {
        let me = Self(ProgressNotifier {
            sender,
            total,
            processed: 0,
            token: WORKSPACE_ANALYSIS_PROGRESS_TOKEN,
            label: "analyzing packages",
        });
        me.0.begin(req_id);
        me
    }

    pub(crate) fn report(&mut self, processed: usize) -> IsDone {
        self.0.report(processed)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct IsDone(pub bool);

#[derive(Debug)]
struct ProgressNotifier {
    sender: Sender<Message>,
    token: &'static str,
    label: &'static str,
    processed: usize,
    total: usize,
}

impl ProgressNotifier {
    fn begin(&self, req_id: RequestId) {
        let create_req = request_new::<lsp_types::request::WorkDoneProgressCreate>(
            req_id,
            WorkDoneProgressCreateParams {
                token: lsp_types::ProgressToken::String(self.token.to_owned()),
            },
        );
        self.sender.send(create_req.into()).unwrap();
        self.send_notification(WorkDoneProgress::Begin(WorkDoneProgressBegin {
            cancellable: None,
            title: "rust-analyzer".to_owned(),
            percentage: Some(self.percentage()),
            message: Some(self.create_progress_message()),
        }));
    }

    fn report(&mut self, processed: usize) -> IsDone {
        if self.processed != processed {
            self.processed = processed;

            self.send_notification(WorkDoneProgress::Report(WorkDoneProgressReport {
                cancellable: None,
                percentage: Some(self.percentage()),
                message: Some(self.create_progress_message()),
            }));
        }
        IsDone(processed >= self.total)
    }

    fn end(&mut self, message: String) {
        self.send_notification(WorkDoneProgress::End(WorkDoneProgressEnd {
            message: Some(message),
        }));
    }

    fn send_notification(&self, progress: WorkDoneProgress) {
        let notif = notification_new::<lsp_types::notification::Progress>(lsp_types::ProgressParams {
            token: lsp_types::ProgressToken::String(self.token.to_owned()),
            value: lsp_types::ProgressParamsValue::WorkDone(progress),
        });
        self.sender.send(notif.into()).unwrap();
    }

    fn create_progress_message(&self) -> String {
        format!("{} ({}/{})", self.label, self.processed, self.total)
    }

    fn percentage(&self) -> f64 {
        (100 * self.processed) as f64 / self.total as f64
    }
}
