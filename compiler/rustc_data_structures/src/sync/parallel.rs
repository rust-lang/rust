//! This module defines parallel operations that are implemented in
//! one way for the serial compiler, and another way the parallel compiler.

#![allow(dead_code)]

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

fn serial_join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
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

pub fn join4<F0, F1, F2, F3, R0, R1, R2, R3>(
    oper0: F0,
    oper1: F1,
    oper2: F2,
    oper3: F3,
) -> (R0, R1, R2, R3)
where
    F0: FnOnce() -> R0 + DynSend,
    F1: FnOnce() -> R1 + DynSend,
    F2: FnOnce() -> R2 + DynSend,
    F3: FnOnce() -> R3 + DynSend,
    R0: DynSend,
    R1: DynSend,
    R2: DynSend,
    R3: DynSend,
{
    if mode::is_dyn_thread_safe() {
        let oper0 = FromDyn::from(oper0);
        let oper1 = FromDyn::from(oper1);
        let oper2 = FromDyn::from(oper2);
        let oper3 = FromDyn::from(oper3);
        // Swap closures around because Chili executes second one on the current thread
        let (r1, (r2, (r3, r0))) = parallel_guard(|guard| {
            let mut scope = chili::Scope::global();
            scope.join_with_heartbeat_every::<1, _, _, _, _>(
                move |_| guard.run(move || FromDyn::from(oper1.into_inner()())),
                move |scope| {
                    scope.join_with_heartbeat_every::<1, _, _, _, _>(
                        move |_| guard.run(move || FromDyn::from(oper2.into_inner()())),
                        move |scope| {
                            scope.join_with_heartbeat_every::<1, _, _, _, _>(
                                move |_| guard.run(move || FromDyn::from(oper3.into_inner()())),
                                move |_| guard.run(move || FromDyn::from(oper0.into_inner()())),
                            )
                        },
                    )
                },
            )
        });
        (
            r0.unwrap().into_inner(),
            r1.unwrap().into_inner(),
            r2.unwrap().into_inner(),
            r3.unwrap().into_inner(),
        )
    } else {
        let (r0, r1, r2, r3) = parallel_guard(|guard| {
            (guard.run(oper0), guard.run(oper1), guard.run(oper2), guard.run(oper3))
        });
        (r0.unwrap(), r1.unwrap(), r2.unwrap(), r3.unwrap())
    }
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
        let (b, a) = parallel_guard(|guard| {
            chili::Scope::global().join_with_heartbeat_every::<1, _, _, _, _>(
                // Swap arguments around because Chili executes second one on the current thread
                move |_| guard.run(move || FromDyn::from(oper_b.into_inner()())),
                move |_| guard.run(move || FromDyn::from(oper_a.into_inner()())),
            )
        });
        (a.unwrap().into_inner(), b.unwrap().into_inner())
    } else {
        serial_join(oper_a, oper_b)
    }
}

fn par_slice<I: DynSend>(
    items: &mut [I],
    guard: &ParallelGuard,
    for_each: impl Fn(&mut I) + DynSync + DynSend,
) {
    struct State<'a, F> {
        for_each: FromDyn<F>,
        guard: &'a ParallelGuard,
        group: usize,
    }

    fn par_rec<I: DynSend, F: Fn(&mut I) + DynSync + DynSend>(
        scope: &mut chili::Scope<'_>,
        items: &mut [I],
        state: &State<'_, F>,
    ) {
        if items.len() <= state.group {
            for item in items {
                state.guard.run(|| (state.for_each)(item));
            }
        } else {
            let (left, right) = items.split_at_mut(items.len() / 2);
            let mut left = state.for_each.derive(left);
            let mut right = state.for_each.derive(right);
            scope.join(
                // Swap arguments around because Chili executes second one on the current thread
                move |scope| par_rec(scope, *right, state),
                move |scope| par_rec(scope, *left, state),
            );
        }
    }

    let mut scope = chili::Scope::global();
    let state = State {
        for_each: FromDyn::from(for_each),
        guard,
        group: std::cmp::max(items.len() / 128, 1),
    };
    par_rec(&mut scope, items, &state)
}

pub fn par_for_each_in<I: DynSend, T: IntoIterator<Item = I>>(
    t: T,
    for_each: impl Fn(&I) + DynSync + DynSend,
) {
    parallel_guard(|guard| {
        if mode::is_dyn_thread_safe() {
            let mut items: Vec<_> = t.into_iter().collect();
            par_slice(&mut items, guard, |i| for_each(&*i))
        } else {
            t.into_iter().for_each(|i| {
                guard.run(|| for_each(&i));
            });
        }
    });
}

/// This runs `for_each` in parallel for each iterator item. If one or more of the
/// `for_each` calls returns `Err`, the function will also return `Err`. The error returned
/// will be non-deterministic, but this is expected to be used with `ErrorGuaranteed` which
/// are all equivalent.
pub fn try_par_for_each_in<T: IntoIterator, E: DynSend>(
    t: T,
    for_each: impl Fn(&<T as IntoIterator>::Item) -> Result<(), E> + DynSync + DynSend,
) -> Result<(), E>
where
    <T as IntoIterator>::Item: DynSend,
{
    parallel_guard(|guard| {
        if mode::is_dyn_thread_safe() {
            let mut items: Vec<_> = t.into_iter().collect();

            let error = Mutex::new(None);

            par_slice(&mut items, guard, |i| {
                if let Err(err) = for_each(&*i) {
                    *error.lock() = Some(err);
                }
            });

            if let Some(err) = error.into_inner() { Err(err) } else { Ok(()) }
        } else {
            t.into_iter().filter_map(|i| guard.run(|| for_each(&i))).fold(Ok(()), Result::and)
        }
    })
}

pub fn par_map<I: DynSend, T: IntoIterator<Item = I>, R: DynSend, C: FromIterator<R>>(
    t: T,
    map: impl Fn(I) -> R + DynSync + DynSend,
) -> C {
    parallel_guard(|guard| {
        if mode::is_dyn_thread_safe() {
            let map = FromDyn::from(map);

            let mut items: Vec<(Option<I>, Option<R>)> =
                t.into_iter().map(|i| (Some(i), None)).collect();

            par_slice(&mut items, guard, |i| {
                i.1 = Some(map(i.0.take().unwrap()));
            });

            items.into_iter().filter_map(|i| i.1).collect()
        } else {
            t.into_iter().filter_map(|i| guard.run(|| map(i))).collect()
        }
    })
}
