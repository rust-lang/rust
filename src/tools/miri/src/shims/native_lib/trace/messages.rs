//! Houses the types that are directly sent across the IPC channels.
//!
//! The overall structure of a traced FFI call, from the child process's POV, is
//! as follows:
//! ```
//! message_tx.send(TraceRequest::StartFfi);
//! confirm_rx.recv();
//! raise(SIGSTOP);
//! /* do ffi call */
//! raise(SIGUSR1); // morally equivalent to some kind of "TraceRequest::EndFfi"
//! let events = event_rx.recv();
//! ```
//! `TraceRequest::OverrideRetcode` can be sent at any point in the above, including
//! before or after all of them.
//!
//! NB: sending these events out of order, skipping steps, etc. will result in
//! unspecified behaviour from the supervisor process, so use the abstractions
//! in `super::child` (namely `start_ffi()` and `end_ffi()`) to handle this. It is
//! trivially easy to cause a deadlock or crash by messing this up!

use std::ops::Range;

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
    OverrideRetcode(i32),
}

/// Information needed to begin tracing.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct StartFfiInfo {
    /// A vector of page addresses. These should have been automatically obtained
    /// with `IsolatedAlloc::pages` and prepared with `IsolatedAlloc::prepare_ffi`.
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

/// The final results of an FFI trace, containing every relevant event detected
/// by the tracer. Sent by the supervisor after receiving a `SIGUSR1` signal.
///
/// The sender for this channel should live on the parent process.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct MemEvents {
    /// An ordered list of memory accesses that occurred. These should be assumed
    /// to be overcautious; that is, if the size of an access is uncertain it is
    /// pessimistically rounded up, and if the type (read/write/both) is uncertain
    /// it is reported as whatever would be safest to assume; i.e. a read + maybe-write
    /// becomes a read + write, etc.
    pub acc_events: Vec<AccessEvent>,
}

/// A single memory access, conservatively overestimated
/// in case of ambiguity.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum AccessEvent {
    /// A read may have occurred on no more than the specified address range.
    Read(Range<usize>),
    /// A write may have occurred on no more than the specified address range.
    Write(Range<usize>),
}
