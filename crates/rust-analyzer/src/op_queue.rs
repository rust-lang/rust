//! Bookkeeping to make sure only one long-running operation is executed.

#[derive(Default)]
pub(crate) struct OpQueue {
    op_scheduled: bool,
    op_in_progress: bool,
}

impl OpQueue {
    pub(crate) fn request_op(&mut self) {
        self.op_scheduled = true;
    }
    pub(crate) fn should_start_op(&mut self) -> bool {
        if !self.op_in_progress && self.op_scheduled {
            self.op_in_progress = true;
            self.op_scheduled = false;
            return true;
        }
        false
    }
    pub(crate) fn op_completed(&mut self) {
        assert!(self.op_in_progress);
        self.op_in_progress = false;
    }
}
