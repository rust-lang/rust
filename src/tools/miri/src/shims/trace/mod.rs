mod child;
pub mod messages;
mod parent;

use std::ops::Range;

pub use self::child::{Supervisor, init_sv, register_retcode_sv};

/// The size used for the array into which we can move the stack pointer.
const FAKE_STACK_SIZE: usize = 1024;

/// Information needed to begin tracing.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct StartFfiInfo {
    /// A vector of page addresses. These should have been automatically obtained
    /// with `IsolatedAlloc::pages` and prepared with `IsolatedAlloc::prepare_ffi`.
    page_ptrs: Vec<usize>,
    /// The address of an allocation that can serve as a temporary stack.
    /// This should be a leaked `Box<[u8; FAKE_STACK_SIZE]>` cast to an int.
    stack_ptr: usize,
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
