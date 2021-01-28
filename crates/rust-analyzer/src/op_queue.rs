//! Bookkeeping to make sure only one long-running operation is executed.

pub(crate) struct OpQueue<D> {
    op_scheduled: Option<D>,
    op_in_progress: bool,
}

impl<D> Default for OpQueue<D> {
    fn default() -> Self {
        Self { op_scheduled: None, op_in_progress: false }
    }
}

impl<D> OpQueue<D> {
    pub(crate) fn request_op(&mut self, data: D) {
        self.op_scheduled = Some(data);
    }
    pub(crate) fn should_start_op(&mut self) -> Option<D> {
        if self.op_in_progress {
            return None;
        }
        self.op_in_progress = self.op_scheduled.is_some();
        self.op_scheduled.take()
    }
    pub(crate) fn op_completed(&mut self) {
        assert!(self.op_in_progress);
        self.op_in_progress = false;
    }
}
