//! This module defines parallel operations that are implemented in
//! one way for the serial compiler, and another way the parallel compiler.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use parking_lot::Mutex;

use crate::FatalErrorMarker;
use crate::sync::{DynSend, DynSync, FromDyn, IntoDynSyncSend, mode};

/// A guard used to hold panics that occur during a parallel section to later by unwound.
/// This is used for the parallel compiler to prevent fatal errors from non-deterministically
/// hiding errors by ensuring that everything in the section has completed executing before
/// continuing with unwinding. It's also used for the non-parallel code to ensure error message
/// output match the parallel compiler for testing purposes.
pub struct ParallelGuard {
    panic: Mutex<Option<IntoDynSyncSend<Box<dyn Any + Send + 'static>>>>,
}

impl ParallelGuard {
    pub fn run<R>(&self, f: impl FnOnce() -> R) -> Option<R> {
        catch_unwind(AssertUnwindSafe(f))
            .map_err(|err| {
                let mut panic = self.panic.lock();
                if panic.is_none() || !(*err).is::<FatalErrorMarker>() {
                    *panic = Some(IntoDynSyncSend(err));
                }
            })
            .ok()
    }
}

/// This gives access to a fresh parallel guard in the closure and will unwind any panics
/// caught in it after the closure returns.
#[inline]
pub fn parallel_guard<R>(f: impl FnOnce(&ParallelGuard) -> R) -> R {
    let guard = ParallelGuard { panic: Mutex::new(None) };
    let ret = f(&guard);
    if let Some(IntoDynSyncSend(panic)) = guard.panic.into_inner() {
        resume_unwind(panic);
    }
    ret
}

pub fn spawn(func: impl FnOnce() + DynSend + 'static) {
    if mode::is_dyn_thread_safe() {
        let func = FromDyn::from(func);
        rustc_thread_pool::spawn(|| {
            (func.into_inner())();
        });
    } else {
        func()
    }
}

pub fn broadcast<R: DynSend>(op: impl Fn(usize) -> R + DynSync) -> Vec<R> {
    if mode::is_dyn_thread_safe() {
        let op = FromDyn::from(op);
        let results = rustc_thread_pool::broadcast(|context| op.derive(op(context.index())));
        results.into_iter().map(|r| r.into_inner()).collect()
    } else {
        vec![op(0)]
    }
}
