use std::cell::RefMut;
use std::collections::BTreeMap;
use std::io;
use std::ops::BitOrAssign;
use std::time::Duration;

use mio::event::Source;
use mio::{Events, Interest, Poll, Token};

use crate::shims::{
    EpollEvalContextExt, FdId, FileDescription, FileDescriptionRef, WeakFileDescriptionRef,
};
use crate::*;

/// Capacity of the event queue which can be polled at a time.
/// Since we don't expect many simultaneous blocking I/O events
/// this value can be set rather low.
const IO_EVENT_CAPACITY: usize = 16;

/// Trait for file descriptions that contain a mio [`Source`].
pub trait SourceFileDescription: FileDescription {
    /// Invoke `f` on the source inside `self`.
    fn with_source(&self, f: &mut dyn FnMut(&mut dyn Source) -> io::Result<()>) -> io::Result<()>;

    /// Get a mutable reference to the readiness of the source.
    fn get_readiness_mut(&self) -> RefMut<'_, BlockingIoSourceReadiness>;
}

/// An I/O interest for a blocked thread. Note that all threads are always considered
/// to be interested in "error" events.
#[derive(Debug, Clone, Copy)]
pub enum BlockingIoInterest {
    /// The blocked thread is interested in [`Interest::READABLE`].
    Read,
    /// The blocked thread is interested in [`Interest::WRITABLE`].
    Write,
    /// The blocked thread is interested in [`Interest::READABLE`] and
    /// [`Interest::WRITABLE`].
    ReadWrite,
}

/// Struct reflecting the readiness of a source file description.
#[derive(Debug)]
pub struct BlockingIoSourceReadiness {
    /// Boolean whether the source is currently readable.
    pub readable: bool,
    /// Boolean whether the source is currently writable.
    pub writable: bool,
    /// Boolean whether the read end of the source has been
    /// closed.
    pub read_closed: bool,
    /// Boolean whether the write end of the source has been
    /// closed.
    pub write_closed: bool,
    /// Boolean whether the source currently has an error.
    pub error: bool,
}

impl BlockingIoSourceReadiness {
    pub fn empty() -> Self {
        Self {
            readable: false,
            writable: false,
            read_closed: false,
            write_closed: false,
            error: false,
        }
    }

    /// Check whether the current readiness fulfills the blocking I/O interest of
    /// `interest`.
    /// This function also returns `true` if the error readiness is set
    /// even when the requested interest might not be fulfilled.
    fn fulfills_interest(&self, interest: &BlockingIoInterest) -> bool {
        match interest {
            BlockingIoInterest::Read => self.readable || self.error,
            BlockingIoInterest::Write => self.writable || self.error,
            BlockingIoInterest::ReadWrite => self.readable || self.writable || self.error,
        }
    }
}

impl BitOrAssign for BlockingIoSourceReadiness {
    fn bitor_assign(&mut self, rhs: Self) {
        self.readable |= rhs.readable;
        self.writable |= rhs.writable;
        self.read_closed |= rhs.read_closed;
        self.write_closed |= rhs.write_closed;
        self.error |= rhs.error;
    }
}

impl From<&mio::event::Event> for BlockingIoSourceReadiness {
    fn from(event: &mio::event::Event) -> Self {
        Self {
            readable: event.is_readable(),
            writable: event.is_writable(),
            read_closed: event.is_read_closed(),
            write_closed: event.is_write_closed(),
            error: event.is_error(),
        }
    }
}

struct BlockingIoSource {
    /// The source file description which is registered into the poll.
    /// We only store weak references such that source file descriptions
    /// can be destroyed whilst they are registered. However, they are required
    /// to deregister themselves when [`FileDescription::destroy`] is called.
    fd: WeakFileDescriptionRef<dyn SourceFileDescription>,
    /// The threads which are blocked on the I/O source, and the interest indicating
    /// when they should be unblocked.
    blocked_threads: BTreeMap<ThreadId, BlockingIoInterest>,
}

/// Manager for managing blocking host I/O in a non-blocking manner.
/// We use [`Poll`] to poll for new I/O events from the OS for sources
/// registered using this manager.
///
/// The semantics of this manager are that host I/O sources are registered
/// to a [`Poll`] for their entire lifespan. Once host readiness events happen
/// on a registered source, its internal epoll readiness gets updated -- even
/// when the source isn't part of an active epoll instance. Also, for the entire
/// lifespan of the source, threads can be added which should be unblocked
/// once a certain [`BlockingIoSourceReadiness`] for an I/O source is satisfied.
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
    /// Map from source file description ids to the actual sources and their
    /// blocked threads.
    sources: BTreeMap<FdId, BlockingIoSource>,
}

impl BlockingIoManager {
    /// Create a new blocking I/O manager instance based on the availability
    /// of communication with the host.
    pub fn new(communicate: bool) -> Result<Self, io::Error> {
        let manager = Self {
            poll: communicate.then_some(Poll::new()?),
            events: Events::with_capacity(IO_EVENT_CAPACITY),
            sources: BTreeMap::default(),
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
    /// The events also immediately get processed: threads get unblocked, and epoll readiness gets updated.
    pub fn poll<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        timeout: Option<Duration>,
    ) -> InterpResult<'tcx, Result<(), io::Error>> {
        let poll = ecx
            .machine
            .blocking_io
            .poll
            .as_mut()
            .expect("Blocking I/O should not be called with isolation enabled");

        // Poll for new I/O events from OS and store them in the events buffer.
        if let Err(err) = poll.poll(&mut ecx.machine.blocking_io.events, timeout) {
            return interp_ok(Err(err));
        };

        let event_fds = ecx
            .machine
            .blocking_io
            .events
            .iter()
            .map(|event| {
                let token = event.token();
                // We know all tokens are valid `FdId`.
                let fd_id = FdId::new_unchecked(token.0);
                let source = ecx
                    .machine
                    .blocking_io
                    .sources
                    .get(&fd_id)
                    .expect("Source should be registered");
                let fd = source.fd.upgrade().expect(
                    "Source file description shouldn't be destroyed whilst being registered",
                );

                assert_eq!(fd.id(), fd_id);
                // Update the readiness of the source.
                *fd.get_readiness_mut() |= BlockingIoSourceReadiness::from(event);
                // Put FD into `event_fds` list.
                fd
            })
            .collect::<Vec<_>>();

        // Update the epoll readiness for all source file descriptions which received an event. Also,
        // unblock the threads which are blocked on such a source and whose interests are now fulfilled.
        for fd in event_fds.into_iter() {
            // Update epoll readiness for the `fd` source.
            ecx.update_epoll_active_events(fd.clone(), false)?;

            let source =
                ecx.machine.blocking_io.sources.get(&fd.id()).expect(
                    "Source file description shouldn't be destroyed whilst being registered",
                );

            // List of all thread id's whose interests are currently fulfilled
            // and which are blocked on the `fd` source. This also includes
            // threads whose interests were already fulfilled before the
            // `poll` invocation.
            let threads = source
                .blocked_threads
                .iter()
                .filter_map(|(thread_id, interest)| {
                    fd.get_readiness_mut().fulfills_interest(interest).then_some(*thread_id)
                })
                .collect::<Vec<_>>();

            // Unblock all threads whose interests are currently fulfilled and
            // which are blocked on the `fd` source.
            threads
                .into_iter()
                .try_for_each(|thread_id| ecx.unblock_thread(thread_id, BlockReason::IO))?;
        }

        interp_ok(Ok(()))
    }

    /// Return whether a source file description is currently registered in the
    /// blocking I/O poll.
    /// This can also be used to check whether a file description is a host
    /// I/O source.
    pub fn contains_source(&self, source_id: &FdId) -> bool {
        self.sources.contains_key(source_id)
    }

    /// Register a source file description to the blocking I/O poll.
    pub fn register(&mut self, source_fd: FileDescriptionRef<dyn SourceFileDescription>) {
        let poll =
            self.poll.as_ref().expect("Blocking I/O should not be called with isolation enabled");

        let id = source_fd.id();
        let token = Token(id.to_usize());

        // All possible interests.
        // We only care about the readable and writable interests because those are the only
        // interests which are available on all platforms. Internally, mio also
        // registers an error interest.
        let interest = Interest::READABLE | Interest::WRITABLE;

        // Treat errors from registering as fatal. On UNIX hosts this can only
        // fail due to system resource errors (e.g. ENOMEM or ENOSPC) or when the source is already registered.
        source_fd
            .with_source(&mut |source| poll.registry().register(source, token, interest))
            .unwrap();

        let source = BlockingIoSource {
            fd: FileDescriptionRef::downgrade(&source_fd),
            blocked_threads: BTreeMap::default(),
        };

        self.sources
            .try_insert(id, source)
            .unwrap_or_else(|_| panic!("Source should not already be registered"));
    }

    /// Deregister a source file description from the blocking I/O poll.
    ///
    /// It's assumed that the file description with id `source_id` is already
    /// removed from the file description table.
    pub fn deregister(&mut self, source_id: FdId, source: impl SourceFileDescription) {
        let poll =
            self.poll.as_ref().expect("Blocking I/O should not be called with isolation enabled");

        let stored_source = self.sources.remove(&source_id).expect("Source should be registered");
        // Ensure that the source file description is already removed from the file
        // description table.
        assert!(
            stored_source.fd.upgrade().is_none(),
            "Sources must only be deregistered when they are destroyed"
        );

        // Because we only store `WeakFileDescriptionRef`s and the `stored_source` file description
        // is already destroyed, the weak reference can no longer be upgraded. Thus, we cannot use
        // it to deregister the source from the poll and instead use the `source` argument to deregister.

        // Treat errors from deregistering as fatal. On UNIX hosts this can only
        // fail due to system resource errors (e.g. ENOMEM or ENOSPC).
        source.with_source(&mut |source| poll.registry().deregister(source)).unwrap();
    }

    /// Add a new blocked thread to a registered source. The thread gets unblocked
    /// once its [`BlockingIoInterest`] is fulfilled when calling
    /// [`BlockingIoManager::poll`].
    ///
    /// It's assumed that the thread of `thread_id` isn't already blocked on
    /// the source with id `source_id` and that this source is currently
    /// registered.
    fn add_blocked_thread(
        &mut self,
        source_id: FdId,
        thread_id: ThreadId,
        interest: BlockingIoInterest,
    ) {
        let source = self.sources.get_mut(&source_id).expect("Source should be registered");

        source
            .blocked_threads
            .try_insert(thread_id, interest)
            .expect("Thread cannot be blocked multiple times on the same source");
    }

    /// Remove a blocked thread from a registered source.
    ///
    /// It's assumed that the thread of `thread_id` is blocked on the
    /// source with id `source_id` and that this source is currently
    /// registered.
    pub fn remove_blocked_thread(&mut self, source_id: FdId, thread_id: ThreadId) {
        let source = self.sources.get_mut(&source_id).expect("Source should be registered");
        source.blocked_threads.remove(&thread_id).expect("Thread should be blocked on source");
    }
}

impl<'tcx> EvalContextExt<'tcx> for MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Block the current thread until some interests on an I/O source
    /// are fulfilled or the optional timeout exceeded.
    /// The callback will be invoked when the thread gets unblocked.
    ///
    /// Note that an error interest is implicitly added to `interest`.
    /// This means that the thread will also be unblocked when the error
    /// readiness gets set for the source even when the requested interest
    /// might not be fulfilled.
    ///
    /// There can also be spurious wake-ups by the OS and thus it's the callers
    /// responsibility to verify that the requested I/O interests are
    /// really ready and to block again if they're not.
    ///
    /// It's the callers responsibility to remove the [`BlockingIoInterest`]
    /// from the blocking I/O manager in the provided callback function.
    #[inline]
    fn block_thread_for_io(
        &mut self,
        source_fd: FileDescriptionRef<dyn SourceFileDescription>,
        interest: BlockingIoInterest,
        timeout: Option<(TimeoutClock, TimeoutAnchor, Duration)>,
        callback: DynUnblockCallback<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // We always have to do this since the thread will de-register itself.
        this.machine.blocking_io.add_blocked_thread(source_fd.id(), this.active_thread(), interest);

        if source_fd.get_readiness_mut().fulfills_interest(&interest) {
            // The requested readiness is currently already fulfilled for the provided source.
            // Instead of actually blocking the thread, we just run the callback function.
            callback.call(this, UnblockKind::Ready)
        } else {
            // The I/O readiness is currently not fulfilled. We block the thread
            // until the readiness is fulfilled and execute the callback then.
            this.block_thread(BlockReason::IO, timeout, callback);
            interp_ok(())
        }
    }
}
