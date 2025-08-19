//! Houses the types that are directly sent across the IPC channels.
//!
//! When forking to initialise the supervisor during `init_sv`, the child raises
//! a `SIGSTOP`; if the parent successfully ptraces the child, it will allow it
//! to resume. Else, the child will be killed by the parent.
//!
//! After initialisation is done, the overall structure of a traced FFI call from
//! the child process's POV is as follows:
//! ```
//! message_tx.send(TraceRequest::StartFfi);
//! confirm_rx.recv(); // receives a `Confirmation`
//! raise(SIGSTOP);
//! /* do ffi call */
//! raise(SIGUSR1); // morally equivalent to some kind of "TraceRequest::EndFfi"
//! let events = event_rx.recv(); // receives a `MemEvents`
//! ```
//! `TraceRequest::OverrideRetcode` can be sent at any point in the above, including
//! before or after all of them. `confirm_rx.recv()` is to be called after, to ensure
//! that the child does not exit before the supervisor has registered the return code.
//!
//! NB: sending these events out of order, skipping steps, etc. will result in
//! unspecified behaviour from the supervisor process, so use the abstractions
//! in `super::child` (namely `do_ffi()`) to handle this. It is
//! trivially easy to cause a deadlock or crash by messing this up!

/// An IPC request sent by the child process to the parent.
///
/// The sender for this channel should live on the child process.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum TraceRequest {
    /// Requests that tracing begins. Following this being sent, the child must
    /// wait to receive a `Confirmation` on the respective channel and then
    /// `raise(SIGSTOP)`.
    ///
    /// To avoid possible issues while allocating memory for IPC channels, ending
    /// the tracing is instead done via `raise(SIGUSR1)`.
    StartFfi(StartFfiInfo),
    /// Manually overrides the code that the supervisor will return upon exiting.
    /// Once set, it is permanent. This can be called again to change the value.
    ///
    /// After sending this, the child must wait to receive a `Confirmation`.
    OverrideRetcode(i32),
}

/// Information needed to begin tracing.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct StartFfiInfo {
    /// A vector of page addresses that store the miri heap which is accessible from C.
    pub page_ptrs: Vec<usize>,
    /// The address of an allocation that can serve as a temporary stack.
    /// This should be a leaked `Box<[u8; CALLBACK_STACK_SIZE]>` cast to an int.
    pub stack_ptr: usize,
}

/// A marker type confirming that the supervisor has received the request to begin
/// tracing and is now waiting for a `SIGSTOP`.
///
/// The sender for this channel should live on the parent process.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Confirmation;
