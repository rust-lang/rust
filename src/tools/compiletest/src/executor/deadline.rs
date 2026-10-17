use std::collections::VecDeque;
use std::sync::mpsc::{self, RecvError, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::executor::{CollectedTest, TestId};

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

    /// All calls to [`Instant::now`] go through this wrapper method.
    /// This makes it easier to find all places that read the current time.
    fn now(&self) -> Instant {
        Instant::now()
    }

    pub(crate) fn push(&mut self, id: TestId, test: &'a CollectedTest, timeout_seconds: u64) {
        let deadline = self.now() + Duration::from_secs(timeout_seconds);
        self.queue.push_back(DeadlineEntry { id, test, deadline });
    }

    /// Equivalent to `rx.recv()`, except that if a test exceeds its deadline
    /// during the wait, the given callback will also be called for that test.
    pub(crate) fn read_channel_while_checking_deadlines<T>(
        &mut self,
        rx: &mpsc::Receiver<T>,
        is_running: impl Fn(TestId) -> bool,
        mut on_deadline_passed: impl FnMut(TestId, &CollectedTest),
    ) -> Result<T, RecvError> {
        loop {
            let Some(next_deadline) = self.next_deadline() else {
                // All currently-running tests have already exceeded their
                // deadline, so do a normal receive.
                return rx.recv();
            };
            let next_deadline_timeout = next_deadline.saturating_duration_since(self.now());

            let recv_result = rx.recv_timeout(next_deadline_timeout);
            // Process deadlines after every receive attempt, regardless of
            // outcome, so that we don't build up an unbounded backlog of stale
            // entries due to a constant stream of tests finishing.
            self.for_each_entry_past_deadline(&is_running, &mut on_deadline_passed);

            match recv_result {
                Ok(value) => return Ok(value),
                // Deadlines have already been processed, so loop and do another receive.
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err(RecvError),
            }
        }
    }

    fn next_deadline(&self) -> Option<Instant> {
        self.queue.iter().map(|entry| entry.deadline).min()
    }

    fn for_each_entry_past_deadline(
        &mut self,
        is_running: impl Fn(TestId) -> bool,
        mut on_deadline_passed: impl FnMut(TestId, &CollectedTest),
    ) {
        let now = self.now();

        // Invoke callbacks for entries past their deadline that are still running.
        for entry in self.queue.iter() {
            if entry.deadline <= now && is_running(entry.id) {
                on_deadline_passed(entry.id, entry.test);
            }
        }

        // Remove entries that are past their deadline or no longer running.
        self.queue.retain(|entry| entry.deadline > now && is_running(entry.id));

        debug_assert!(self.queue.iter().all(|entry| now < entry.deadline));
    }
}
