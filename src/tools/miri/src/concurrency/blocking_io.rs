use std::io;
use std::time::Duration;

use mio::event::Source;
use mio::{Events, Interest, Poll, Token};
use rustc_data_structures::fx::FxHashMap;

use crate::shims::FdId;
use crate::*;

/// Capacity of the event queue which can be polled at a time.
/// Since we don't expect many simultaneous blocking I/O events
/// this value can be set rather low.
const IO_EVENT_CAPACITY: usize = 16;

/// Trait for values that contain a mio [`Source`].
pub trait WithSource {
    /// Invoke `f` on the source inside `self`.
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()>;

    /// Unique identifier of the file description to which the
    /// source belongs.
    fn id(&self) -> FdId;
}

#[derive(Debug, Hash, PartialEq, Clone, Copy, Eq, PartialOrd, Ord)]
pub enum InterestReason {
    /// A thread is blocked on this source and wants to get unblocked
    /// when the interest on the source is fulfilled.
    ThreadBlocked(ThreadId),
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
    /// Map from source ids to the actual sources and the registered interests for
    /// every source.
    sources: FxHashMap<FdId, (Box<dyn WithSource>, FxHashMap<InterestReason, Interest>)>,
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
    /// Returns all interest reasons which are fulfilled through an I/O event together with the
    /// sources they are registered for.
    pub fn poll(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Vec<(InterestReason, FdId)>, io::Error> {
        let poll =
            self.poll.as_mut().expect("Blocking I/O should not be called with isolation enabled");

        // Poll for new I/O events from OS and store them in the events buffer.
        poll.poll(&mut self.events, timeout)?;

        let ready = self
            .events
            .iter()
            .flat_map(|event| {
                let token = event.token();
                let fd_id = FdId::new_unchecked(token.0);
                let (_, interests) = self.sources.get(&fd_id).expect("Source should be registered");
                interests
                    .iter()
                    .filter(|(_, interest)| {
                        // Only retain interests of the source which are fulfilled by
                        // the current event.
                        (event.is_readable() && interest.is_readable())
                            || (event.is_writable() && interest.is_writable())
                            || (event.is_aio() && interest.is_aio())
                            || (event.is_lio() && interest.is_lio())
                            || (event.is_priority() && interest.is_priority())
                    })
                    .map(move |(reason, _)| (*reason, fd_id))
            })
            .collect::<Vec<_>>();

        Ok(ready)
    }

    /// Register an interest for a blocking I/O source.
    ///
    /// As the OS can always produce spurious wake-ups, it's the callers responsibility to
    /// verify the requested I/O interests are really ready and to register again if they're not.
    ///
    /// It's assumed that no interest is already registered for this source with the same reason!
    pub fn register(
        &mut self,
        source: Box<dyn WithSource>,
        reason: InterestReason,
        interest: Interest,
    ) {
        let poll =
            self.poll.as_ref().expect("Blocking I/O should not be called with isolation enabled");

        let id = source.id();
        let token = Token(id.to_usize());

        let Some((_, current_interests)) = self.sources.get_mut(&id) else {
            // Treat errors from registering as fatal. On UNIX hosts this can only
            // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
            source
                .with_source(&mut |source| source.register(poll.registry(), token, interest))
                .unwrap();

            self.sources.insert(id, (source, FxHashMap::from_iter([(reason, interest)])));
            return;
        };

        // The source is already registered. We need to check whether we need to
        // reregister because the provided interest contains new interests for the source.

        let old_interest =
            interest_union(current_interests).expect("Source should contain at least one interest");

        current_interests
            .try_insert(reason, interest)
            .unwrap_or_else(|_| panic!("Interest reason should be unique"));

        let new_interest = old_interest.add(interest);
        if new_interest == old_interest {
            // The overall interests in the source did not change and thus we
            // don't need to reregister the source.
            return;
        }

        // The overall interests in the source changed. We need to reregister
        // the source with the updated interests.

        // Treat errors from reregistering as fatal. On UNIX hosts this can only
        // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
        source
            .with_source(&mut |source| source.reregister(poll.registry(), token, new_interest))
            .unwrap();
    }

    /// Deregister an interest from a blocking I/O source.
    ///
    /// The source and the interest on this source are assumed to be registered!
    pub fn deregister(&mut self, source_id: FdId, reason: InterestReason) {
        let poll =
            self.poll.as_ref().expect("Blocking I/O should not be called with isolation enabled");

        let token = Token(source_id.to_usize());
        let (source, current_interests) =
            self.sources.get_mut(&source_id).expect("Source should be registered");
        let old_interest =
            interest_union(current_interests).expect("Source should contain at least one interest");

        current_interests
            .remove(&reason)
            .unwrap_or_else(|| panic!("Interest reason should be registered for source"));

        let Some(new_interest) = interest_union(current_interests) else {
            // There are no longer any interests in this source.
            // We can thus deregister the source from the poll.

            // Treat errors from deregistering as fatal. On UNIX hosts this can only
            // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
            source.with_source(&mut |source| source.deregister(poll.registry())).unwrap();
            self.sources.remove(&source_id);
            return;
        };

        if new_interest == old_interest {
            // The overall interests in the source did not change and thus we
            // don't need to reregister the source.
            return;
        }

        // The overall interests in the source changed. We need to reregister
        // the source with the updated interests.

        // Treat errors from reregistering as fatal. On UNIX hosts this can only
        // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
        source
            .with_source(&mut |source| source.reregister(poll.registry(), token, new_interest))
            .unwrap();
    }
}

/// Get the union of all interests for a source.
fn interest_union(interests: &FxHashMap<InterestReason, Interest>) -> Option<Interest> {
    interests
        .values()
        .copied()
        .fold(None, |acc, interest| acc.map(|acc: Interest| acc.add(interest)).or(Some(interest)))
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
            InterestReason::ThreadBlocked(this.machine.threads.active_thread()),
            interests,
        );
        this.block_thread(BlockReason::IO, timeout, callback);
    }
}
