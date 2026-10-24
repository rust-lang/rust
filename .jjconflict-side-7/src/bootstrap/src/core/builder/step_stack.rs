use std::cell::RefCell;

thread_local! {
    static STEP_STACK: RefCell<StepStack> = const { RefCell::new(StepStack::new()) };
}

/// This type serves for recording the stack of executed steps in a thread-local variable.
///
/// It is used to print the currently running step stack in places where we do not have easy
/// access to the currently used builder, e.g. in exit! macros or the panic handler.
pub struct StepStack {
    stack: Vec<StepRecord>,
}

pub struct StepRecord {
    pub info: String,
    pub location: String,
}

impl StepStack {
    /// Return the currently active step stack for this thread.
    pub fn with_current<F>(func: F)
    where
        F: FnOnce(&mut StepStack),
    {
        STEP_STACK.with(|stack| func(&mut stack.borrow_mut()));
    }

    const fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn get_active_steps(&self) -> impl Iterator<Item = &StepRecord> {
        self.stack.iter()
    }

    pub fn clear(&mut self) {
        self.stack.clear();
    }

    pub fn push(&mut self, record: StepRecord) {
        self.stack.push(record);
    }

    pub fn pop(&mut self) {
        self.stack.pop();
    }
}
