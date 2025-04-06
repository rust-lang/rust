use std::collections::VecDeque;
use std::sync::mpsc::{self, RecvError, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::executor::{CollectedTest, TestId};

const TEST_WARN_TIMEOUT_S: u64 = 60;

struct DeadlineEntry<'a> {
    id: TestId,
    test: &'a CollectedTest,
    deadline: Instant,
}

pub(crate) struct DeadlineQueue<'a> {
    queue: VecDeque<DeadlineEntry<'a>>,
}

impl<'a> DeadlineQueue<'a> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { queue: VecDeque::with_capacity(capacity) }
    }

    pub(crate) fn push(&mut self, id: TestId, test: &'a CollectedTest) {
        let deadline = Instant::now() + Duration::from_secs(TEST_WARN_TIMEOUT_S);
        self.queue.push_back(DeadlineEntry { id, test, deadline });
    }

    /// Equivalent to `rx.read()`, except that if any test exceeds its deadline
    /// during the wait, the given callback will also be called for that test.
    pub(crate) fn read_channel_while_checking_deadlines<T>(
        &mut self,
        rx: &mpsc::Receiver<T>,
        mut on_deadline_passed: impl FnMut(TestId, &CollectedTest),
    ) -> Result<T, RecvError> {
        loop {
            let Some(next_deadline) = self.next_deadline() else {
                // All currently-running tests have already exceeded their
                // deadline, so do a normal receive.
                return rx.recv();
            };
            let wait_duration = next_deadline.saturating_duration_since(Instant::now());

            let recv_result = rx.recv_timeout(wait_duration);
            match recv_result {
                Ok(value) => return Ok(value),
                Err(RecvTimeoutError::Timeout) => {
                    // Notify the callback of tests that have exceeded their
                    // deadline, then loop and do annother channel read.
                    for DeadlineEntry { id, test, .. } in self.remove_tests_past_deadline() {
                        on_deadline_passed(id, test);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Err(RecvError),
            }
        }
    }

    fn next_deadline(&self) -> Option<Instant> {
        Some(self.queue.front()?.deadline)
    }

    fn remove_tests_past_deadline(&mut self) -> Vec<DeadlineEntry<'a>> {
        let now = Instant::now();
        let mut timed_out = vec![];
        while let Some(deadline_entry) = pop_front_if(&mut self.queue, |entry| now < entry.deadline)
        {
            timed_out.push(deadline_entry);
        }
        timed_out
    }
}

/// FIXME(vec_deque_pop_if): Use `VecDeque::pop_front_if` when it is stable in bootstrap.
fn pop_front_if<T>(queue: &mut VecDeque<T>, predicate: impl FnOnce(&T) -> bool) -> Option<T> {
    let first = queue.front()?;
    if predicate(first) { queue.pop_front() } else { None }
}
