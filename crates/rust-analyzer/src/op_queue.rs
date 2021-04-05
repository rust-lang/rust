//! Bookkeeping to make sure only one long-running operation is being executed
//! at a time.

pub(crate) struct OpQueue<Args, Output> {
    op_scheduled: Option<Args>,
    op_in_progress: bool,
    last_op_result: Output,
}

impl<Args, Output: Default> Default for OpQueue<Args, Output> {
    fn default() -> Self {
        Self { op_scheduled: None, op_in_progress: false, last_op_result: Default::default() }
    }
}

impl<Args, Output> OpQueue<Args, Output> {
    pub(crate) fn request_op(&mut self, data: Args) {
        self.op_scheduled = Some(data);
    }
    pub(crate) fn should_start_op(&mut self) -> Option<Args> {
        if self.op_in_progress {
            return None;
        }
        self.op_in_progress = self.op_scheduled.is_some();
        self.op_scheduled.take()
    }
    pub(crate) fn op_completed(&mut self, result: Output) {
        assert!(self.op_in_progress);
        self.op_in_progress = false;
        self.last_op_result = result;
    }

    #[allow(unused)]
    pub(crate) fn last_op_result(&self) -> &Output {
        &self.last_op_result
    }
}
