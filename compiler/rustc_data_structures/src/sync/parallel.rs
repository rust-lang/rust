//! This module defines parallel operations that are implemented in
//! one way for the serial compiler, and another way the parallel compiler.

#![allow(dead_code)]

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use parking_lot::Mutex;
use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelIterator};

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

pub fn serial_join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    let (a, b) = parallel_guard(|guard| {
        let a = guard.run(oper_a);
        let b = guard.run(oper_b);
        (a, b)
    });
    (a.unwrap(), b.unwrap())
}

/// Runs a list of blocks in parallel. The first block is executed immediately on
/// the current thread. Use that for the longest running block.
#[macro_export]
macro_rules! parallel {
        (impl $fblock:block [$($c:expr,)*] [$block:expr $(, $rest:expr)*]) => {
            parallel!(impl $fblock [$block, $($c,)*] [$($rest),*])
        };
        (impl $fblock:block [$($blocks:expr,)*] []) => {
            $crate::sync::parallel_guard(|guard| {
                $crate::sync::scope(|s| {
                    $(
                        let block = $crate::sync::FromDyn::from(|| $blocks);
                        s.spawn(move |_| {
                            guard.run(move || block.into_inner()());
                        });
                    )*
                    guard.run(|| $fblock);
                });
            });
        };
        ($fblock:block, $($blocks:block),*) => {
            if $crate::sync::is_dyn_thread_safe() {
                // Reverse the order of the later blocks since Rayon executes them in reverse order
                // when using a single thread. This ensures the execution order matches that
                // of a single threaded rustc.
                parallel!(impl $fblock [] [$($blocks),*]);
            } else {
                $crate::sync::parallel_guard(|guard| {
                    guard.run(|| $fblock);
                    $(guard.run(|| $blocks);)*
                });
            }
        };
    }

// This function only works when `mode::is_dyn_thread_safe()`.
pub fn scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&rayon::Scope<'scope>) -> R + DynSend,
    R: DynSend,
{
    let op = FromDyn::from(op);
    rayon::scope(|s| FromDyn::from(op.into_inner()(s))).into_inner()
}

#[inline]
pub fn join<A, B, RA: DynSend, RB: DynSend>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + DynSend,
    B: FnOnce() -> RB + DynSend,
{
    if mode::is_dyn_thread_safe() {
        let oper_a = FromDyn::from(oper_a);
        let oper_b = FromDyn::from(oper_b);
        let (a, b) = parallel_guard(|guard| {
            rayon::join(
                move || guard.run(move || FromDyn::from(oper_a.into_inner()())),
                move || guard.run(move || FromDyn::from(oper_b.into_inner()())),
            )
        });
        (a.unwrap().into_inner(), b.unwrap().into_inner())
    } else {
        serial_join(oper_a, oper_b)
    }
}

pub fn par_for_each_in<I, T: IntoIterator<Item = I> + IntoParallelIterator<Item = I>>(
    t: T,
    for_each: impl Fn(I) + DynSync + DynSend,
) {
    parallel_guard(|guard| {
        if mode::is_dyn_thread_safe() {
            let for_each = FromDyn::from(for_each);
            t.into_par_iter().for_each(|i| {
                guard.run(|| for_each(i));
            });
        } else {
            t.into_iter().for_each(|i| {
                guard.run(|| for_each(i));
            });
        }
    });
}

pub fn try_par_for_each_in<
    T: IntoIterator + IntoParallelIterator<Item = <T as IntoIterator>::Item>,
    E: Send,
>(
    t: T,
    for_each: impl Fn(<T as IntoIterator>::Item) -> Result<(), E> + DynSync + DynSend,
) -> Result<(), E> {
    parallel_guard(|guard| {
        if mode::is_dyn_thread_safe() {
            let for_each = FromDyn::from(for_each);
            t.into_par_iter()
                .filter_map(|i| guard.run(|| for_each(i)))
                .reduce(|| Ok(()), Result::and)
        } else {
            t.into_iter().filter_map(|i| guard.run(|| for_each(i))).fold(Ok(()), Result::and)
        }
    })
}

pub fn par_map<
    I,
    T: IntoIterator<Item = I> + IntoParallelIterator<Item = I>,
    R: std::marker::Send,
    C: FromIterator<R> + FromParallelIterator<R>,
>(
    t: T,
    map: impl Fn(I) -> R + DynSync + DynSend,
) -> C {
    parallel_guard(|guard| {
        if mode::is_dyn_thread_safe() {
            let map = FromDyn::from(map);
            t.into_par_iter().filter_map(|i| guard.run(|| map(i))).collect()
        } else {
            t.into_iter().filter_map(|i| guard.run(|| map(i))).collect()
        }
    })
}
