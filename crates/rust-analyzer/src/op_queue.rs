//! Bookkeeping to make sure only one long-running operation is being executed
//! at a time.

pub(crate) type Cause = String;

pub(crate) struct OpQueue<Args = (), Output = ()> {
    op_requested: Option<(Cause, Args)>,
    op_in_progress: bool,
    last_op_result: Output,
}

impl<Args, Output: Default> Default for OpQueue<Args, Output> {
    fn default() -> Self {
        Self { op_requested: None, op_in_progress: false, last_op_result: Default::default() }
    }
}

impl<Args, Output> OpQueue<Args, Output> {
    pub(crate) fn request_op(&mut self, reason: Cause, args: Args) {
        self.op_requested = Some((reason, args));
    }
    pub(crate) fn should_start_op(&mut self) -> Option<(Cause, Args)> {
        if self.op_in_progress {
            return None;
        }
        self.op_in_progress = self.op_requested.is_some();
        self.op_requested.take()
    }
    pub(crate) fn op_completed(&mut self, result: Output) {
        assert!(self.op_in_progress);
        self.op_in_progress = false;
        self.last_op_result = result;
    }

    pub(crate) fn last_op_result(&self) -> &Output {
        &self.last_op_result
    }
    pub(crate) fn op_in_progress(&self) -> bool {
        self.op_in_progress
    }
    pub(crate) fn op_requested(&self) -> bool {
        self.op_requested.is_some()
    }
}
