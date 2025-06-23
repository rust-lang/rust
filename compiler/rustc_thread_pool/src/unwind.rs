//! Package up unwind recovery. Note that if you are in some sensitive
//! place, you can use the `AbortIfPanic` helper to protect against
//! accidental panics in the rayon code itself.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::thread;

/// Executes `f` and captures any panic, translating that panic into a
/// `Err` result. The assumption is that any panic will be propagated
/// later with `resume_unwinding`, and hence `f` can be treated as
/// exception safe.
pub(super) fn halt_unwinding<F, R>(func: F) -> thread::Result<R>
where
    F: FnOnce() -> R,
{
    panic::catch_unwind(AssertUnwindSafe(func))
}

pub(super) fn resume_unwinding(payload: Box<dyn Any + Send>) -> ! {
    panic::resume_unwind(payload)
}

pub(super) struct AbortIfPanic;

impl Drop for AbortIfPanic {
    fn drop(&mut self) {
        eprintln!("Rayon: detected unexpected panic; aborting");
        ::std::process::abort();
    }
}
