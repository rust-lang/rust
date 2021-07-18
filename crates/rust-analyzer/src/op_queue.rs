//! Bookkeeping to make sure only one long-running operation is being executed
//! at a time.

pub(crate) struct OpQueue<Output> {
    op_requested: bool,
    op_in_progress: bool,
    last_op_result: Output,
}

impl<Output: Default> Default for OpQueue<Output> {
    fn default() -> Self {
        Self { op_requested: false, op_in_progress: false, last_op_result: Default::default() }
    }
}

impl<Output> OpQueue<Output> {
    pub(crate) fn request_op(&mut self) {
        self.op_requested = true;
    }
    pub(crate) fn should_start_op(&mut self) -> bool {
        if self.op_in_progress {
            return false;
        }
        self.op_in_progress = self.op_requested;
        self.op_requested = false;
        self.op_in_progress
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
        self.op_requested
    }
}
