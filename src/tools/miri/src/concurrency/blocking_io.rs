use std::io;
use std::time::Duration;

use mio::event::Source;
use mio::{Events, Interest, Poll, Token};
use rustc_data_structures::fx::FxHashMap;

use crate::*;

/// Capacity of the event queue which can be polled at a time.
/// Since we don't expect many simultaneous blocking I/O events
/// this value can be set rather low.
const IO_EVENT_CAPACITY: usize = 16;

/// Trait for values that contain a mio [`Source`].
pub trait WithSource {
    /// Invoke `f` on the source inside `self`.
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()>;
}

/// Manager for managing blocking host I/O in a non-blocking manner.
/// We use [`Poll`] to poll for new I/O events from the OS for sources
/// registered using this manager.
///
/// Since blocking host I/O is inherently non-deterministic, no method on this
/// manager should be called when isolation is enabled. The only exception is
/// the [`BlockingIoManager::new`] function to create the manager. Everywhere else,
/// we assert that isolation is disabled!
pub struct BlockingIoManager {
    /// Poll instance to monitor I/O events from the OS.
    /// This is only [`None`] when Miri is run with isolation enabled.
    poll: Option<Poll>,
    /// Buffer used to store the ready I/O events when calling [`Poll::poll`].
    /// This is not part of the state and only stored to avoid allocating a
    /// new buffer for every poll.
    events: Events,
    /// Map between threads which are currently blocked and the
    /// underlying I/O source.
    sources: FxHashMap<ThreadId, Box<dyn WithSource>>,
}

impl BlockingIoManager {
    /// Create a new blocking I/O manager instance based on the availability
    /// of communication with the host.
    pub fn new(communicate: bool) -> Result<Self, io::Error> {
        let manager = Self {
            poll: communicate.then_some(Poll::new()?),
            events: Events::with_capacity(IO_EVENT_CAPACITY),
            sources: FxHashMap::default(),
        };
        Ok(manager)
    }

    /// Poll for new I/O events from the OS or wait until the timeout expired.
    ///
    /// - If the timeout is [`Some`] and contains [`Duration::ZERO`], the poll doesn't block and just
    ///   reads all events since the last poll.
    /// - If the timeout is [`Some`] and contains a non-zero duration, it blocks at most for the
    ///   specified duration.
    /// - If the timeout is [`None`] the poll blocks indefinitely until an event occurs.
    ///
    /// Returns all threads that are ready because they received an I/O event.
    pub fn poll(&mut self, timeout: Option<Duration>) -> Result<Vec<ThreadId>, io::Error> {
        let poll =
            self.poll.as_mut().expect("Blocking I/O should not be called with isolation enabled");

        // Poll for new I/O events from OS and store them in the events buffer.
        poll.poll(&mut self.events, timeout)?;

        let ready = self
            .events
            .iter()
            .map(|event| {
                let token = event.token();
                ThreadId::new_unchecked(token.0.try_into().unwrap())
            })
            .collect::<Vec<_>>();

        // Deregister all ready sources as we only want to receive one event per thread.
        ready.iter().for_each(|thread_id| self.deregister(*thread_id));

        Ok(ready)
    }

    /// Register a blocking I/O source for a thread together with it's poll interests.
    ///
    /// The source will be deregistered automatically once an event for it is received.
    ///
    /// As the OS can always produce spurious wake-ups, it's the callers responsibility to
    /// verify the requested I/O interests are really ready and to register again if they're not.
    pub fn register(&mut self, source: Box<dyn WithSource>, thread: ThreadId, interests: Interest) {
        let poll =
            self.poll.as_ref().expect("Blocking I/O should not be called with isolation enabled");

        let token = Token(thread.to_u32().to_usize());

        // Treat errors from registering as fatal. On UNIX hosts this can only
        // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
        source
            .with_source(&mut |source| source.register(poll.registry(), token, interests))
            .unwrap();
        self.sources
            .try_insert(thread, source)
            .unwrap_or_else(|_| panic!("A thread cannot be registered twice at the same time"));
    }

    /// Deregister the event source for a thread. Returns the kind of I/O the thread was
    /// blocked on.
    fn deregister(&mut self, thread: ThreadId) {
        let poll =
            self.poll.as_ref().expect("Blocking I/O should not be called with isolation enabled");

        let Some(source) = self.sources.remove(&thread) else {
            panic!("Attempt to deregister a token which isn't registered")
        };

        // Treat errors from deregistering as fatal. On UNIX hosts this can only
        // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
        source.with_source(&mut |source| source.deregister(poll.registry())).unwrap();
    }
}

impl<'tcx> EvalContextExt<'tcx> for MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Block the current thread until some interests on an I/O source
    /// are fulfilled or the optional timeout exceeded.
    /// The callback will be invoked when the thread gets unblocked.
    ///
    /// There can be spurious wake-ups by the OS and thus it's the callers
    /// responsibility to verify that the requested I/O interests are
    /// really ready and to block again if they're not.
    #[inline]
    fn block_thread_for_io(
        &mut self,
        source: impl WithSource + 'static,
        interests: Interest,
        timeout: Option<(TimeoutClock, TimeoutAnchor, Duration)>,
        callback: DynUnblockCallback<'tcx>,
    ) {
        let this = self.eval_context_mut();
        this.machine.blocking_io.register(
            Box::new(source),
            this.machine.threads.active_thread(),
            interests,
        );
        this.block_thread(BlockReason::IO, timeout, callback);
    }
}
