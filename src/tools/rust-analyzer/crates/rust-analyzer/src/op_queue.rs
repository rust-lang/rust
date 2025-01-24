//! Bookkeeping to make sure only one long-running operation is being executed
//! at a time.

pub(crate) type Cause = String;

/// A single-item queue that allows callers to request an operation to
/// be performed later.
///
/// ```
/// let queue = OpQueue::default();
///
/// // Request work to be done.
/// queue.request_op("user pushed a button", ());
///
/// // In a later iteration of the server loop, we start the work.
/// if let Some((_cause, ())) = queue.should_start_op() {
///     dbg!("Some slow operation here");
/// }
///
/// // In an even later iteration of the server loop, we can see that the work
/// // was completed.
/// if !queue.op_in_progress() {
///     dbg!("Work has been done!");
/// }
/// ```
#[derive(Debug)]
pub(crate) struct OpQueue<Args = (), Output = ()> {
    op_requested: Option<(Cause, Args)>,
    op_in_progress: bool,
    last_op_result: Option<Output>,
}

impl<Args, Output> Default for OpQueue<Args, Output> {
    fn default() -> Self {
        Self { op_requested: None, op_in_progress: false, last_op_result: None }
    }
}

impl<Args, Output> OpQueue<Args, Output> {
    /// Request an operation to start.
    pub(crate) fn request_op(&mut self, reason: Cause, args: Args) {
        self.op_requested = Some((reason, args));
    }

    /// If there was an operation requested, mark this queue as
    /// started and return the request arguments.
    pub(crate) fn should_start_op(&mut self) -> Option<(Cause, Args)> {
        if self.op_in_progress {
            return None;
        }
        self.op_in_progress = self.op_requested.is_some();
        self.op_requested.take()
    }

    /// Mark an operation as completed.
    pub(crate) fn op_completed(&mut self, result: Output) {
        assert!(self.op_in_progress);
        self.op_in_progress = false;
        self.last_op_result = Some(result);
    }

    /// Get the result of the last operation.
    pub(crate) fn last_op_result(&self) -> Option<&Output> {
        self.last_op_result.as_ref()
    }

    // Is there an operation that has started, but hasn't yet finished?
    pub(crate) fn op_in_progress(&self) -> bool {
        self.op_in_progress
    }

    // Has an operation been requested?
    pub(crate) fn op_requested(&self) -> bool {
        self.op_requested.is_some()
    }
}
