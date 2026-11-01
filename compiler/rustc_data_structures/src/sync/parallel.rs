//! This module defines parallel operations that are implemented in
//! one way for the serial compiler, and another way the parallel compiler.

use std::any::Any;
use std::ops::{Index, Range};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use parking_lot::Mutex;

use crate::FatalErrorMarker;
use crate::sync::{DynSend, DynSync, FromDyn, IntoDynSyncSend, mode};

pub trait Dss = DynSend + DynSync;

pub trait HasLength {
    fn len(&self) -> usize;
}

impl<T> HasLength for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> HasLength for indexmap::set::Slice<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

pub trait ParSlice<T> = HasLength + Index<usize, Output = T> + DynSync;

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

pub fn spawn(func: impl FnOnce() + DynSend + 'static) {
    if let Some(proof) = mode::check_dyn_thread_safe() {
        let func = proof.derive(func);
        rustc_thread_pool::spawn(|| {
            (func.into_inner())();
        });
    } else {
        func()
    }
}

/// Runs the functions in parallel.
///
/// The first function is executed immediately on the current thread.
/// Use that for the longest running function for better scheduling.
pub fn par_fns(funcs: &mut [&mut (dyn FnMut() + DynSend)]) {
    parallel_guard(|guard: &ParallelGuard| {
        if let Some(proof) = mode::check_dyn_thread_safe() {
            let funcs = proof.derive(funcs);
            rustc_thread_pool::scope(|s| {
                let Some((first, rest)) = funcs.into_inner().split_at_mut_checked(1) else {
                    return;
                };

                // Reverse the order of the later functions since Rayon executes them in reverse
                // order when using a single thread. This ensures the execution order matches
                // that of a single threaded rustc.
                for f in rest.iter_mut().rev() {
                    let f = proof.derive(f);
                    s.spawn(|_| {
                        guard.run(|| (f.into_inner())());
                    });
                }

                // Run the first function without spawning to
                // ensure it executes immediately on this thread.
                guard.run(|| first[0]());
            });
        } else {
            for f in funcs {
                guard.run(|| f());
            }
        }
    });
}

#[inline]
pub fn par_join<A, B, RA: DynSend, RB: DynSend>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + DynSend,
    B: FnOnce() -> RB + DynSend,
{
    if let Some(proof) = mode::check_dyn_thread_safe() {
        let oper_a = proof.derive(oper_a);
        let oper_b = proof.derive(oper_b);
        let (a, b) = parallel_guard(|guard| {
            rustc_thread_pool::join(
                move || guard.run(move || proof.derive(oper_a.into_inner()())),
                move || guard.run(move || proof.derive(oper_b.into_inner()())),
            )
        });
        (a.unwrap().into_inner(), b.unwrap().into_inner())
    } else {
        serial_join(oper_a, oper_b)
    }
}

const MAX_GROUP_COUNT: usize = 128;

fn par_range_internal(
    range: Range<usize>,
    guard: &ParallelGuard,
    proof: FromDyn<()>,
    op: impl Fn(usize) + Dss,
) {
    let len = range.end - range.start;
    match len {
        0 => return,
        1 => {
            guard.run(|| op(range.start));
            return;
        }
        _ => {}
    }

    let op = proof.derive(op);

    rustc_thread_pool::scope(|s| {
        let group_size = len.div_ceil(MAX_GROUP_COUNT);
        debug_assert!(group_size > 0);

        // Take the first group and execute it after the loop on this thread.
        let first_group = range.start..range.end.min(range.start + group_size);

        let range = first_group.end..range.end;
        let mut curr_end = range.end;

        // We iterate from end to start to reverse the chunks as it is done in `par_slice_mut`.
        loop {
            let curr_start = curr_end.checked_sub(group_size).unwrap_or(range.start);

            // Don't let the `curr_start` go before the initial range start.
            let curr_start = curr_start.max(range.start);

            // We can encounter empty range here, don't spawn anything in this case.
            if curr_start == curr_end {
                break;
            }

            let curr_range = curr_start..curr_end;
            s.spawn(|_| {
                for idx in curr_range {
                    guard.run(|| op(idx));
                }
            });

            if curr_start == range.start {
                break;
            }

            curr_end -= group_size;
        }

        for idx in first_group {
            guard.run(|| op(idx));
        }
    });
}

/// Executes the given action `op` with each index from range `range` in parallel
/// if the parallel frontend is enabled, otherwise executes them sequentially. Action
/// is executed with each index under parallel guard in both scenarios so if one of them
/// panics all others will be executed anyway.
pub fn par_range(range: Range<usize>, op: impl Fn(usize) + Dss) {
    parallel_guard(|guard| {
        if let Some(proof) = mode::check_dyn_thread_safe() {
            par_range_internal(range, guard, proof, op);
        } else {
            for i in range {
                guard.run(|| op(i));
            }
        }
    })
}

fn par_slice_mut<I: DynSend>(
    items: &mut [I],
    guard: &ParallelGuard,
    for_each: impl Fn(&mut I) + Dss,
    proof: FromDyn<()>,
) {
    match items {
        [] => return,
        [item] => {
            guard.run(|| for_each(item));
            return;
        }
        _ => (),
    }

    let for_each = proof.derive(for_each);
    let mut items = for_each.derive(items);
    rustc_thread_pool::scope(|s| {
        let proof = items.derive(());

        let group_size = items.len().div_ceil(MAX_GROUP_COUNT);
        let mut groups = items.chunks_mut(group_size);

        let Some(first_group) = groups.next() else { return };

        // Reverse the order of the later functions since Rayon executes them in reverse
        // order when using a single thread. This ensures the execution order matches
        // that of a single threaded rustc.
        for group in groups.rev() {
            let group = proof.derive(group);
            s.spawn(|_| {
                let mut group = group;
                for i in group.iter_mut() {
                    guard.run(|| for_each(i));
                }
            });
        }

        // Run the first function without spawning to avoid overwhelming stealing.
        for i in first_group.iter_mut() {
            guard.run(|| for_each(i));
        }
    });
}

pub fn par_for_each_in<T: Dss>(items: &(impl ParSlice<T> + ?Sized), for_each: impl Fn(&T) + Dss) {
    let range = 0..items.len();
    parallel_guard(|guard| {
        if let Some(proof) = mode::check_dyn_thread_safe() {
            par_range_internal(range, guard, proof, |idx| for_each(&items[idx]));
        } else {
            for idx in range {
                guard.run(|| for_each(&items[idx]));
            }
        }
    })
}

pub fn par_for_each_slice<T: DynSend>(
    items: &mut [T],
    for_each: impl Fn(&mut T) + DynSync + DynSend,
) {
    parallel_guard(|guard| {
        if let Some(proof) = mode::check_dyn_thread_safe() {
            par_slice(items, guard, |i| for_each(i), proof)
        } else {
            items.iter_mut().for_each(|i| {
                guard.run(|| for_each(i));
            });
        }
    });
}

/// This runs `for_each` in parallel for each iterator item. If one or more of the
/// `for_each` calls returns `Err`, the function will also return `Err`. The error returned
/// will be non-deterministic, but this is expected to be used with `ErrorGuaranteed` which
/// are all equivalent.
pub fn try_par_for_each_in<T: Dss, E: DynSend>(
    items: &(impl ParSlice<T> + ?Sized),
    for_each: impl Fn(&T) -> Result<(), E> + Dss,
) -> Result<(), E> {
    let range = 0..items.len();
    parallel_guard(|guard| {
        if let Some(proof) = mode::check_dyn_thread_safe() {
            let error = Mutex::new(None);

            par_range_internal(range, guard, proof, |idx| {
                if let Err(err) = for_each(&items[idx]) {
                    *error.lock() = Some(err);
                }
            });

            if let Some(err) = error.into_inner() { Err(err) } else { Ok(()) }
        } else {
            range
                .into_iter()
                .filter_map(|idx| guard.run(|| for_each(&items[idx])))
                .fold(Ok(()), Result::and)
        }
    })
}

pub fn par_map<I: DynSend, T: IntoIterator<Item = I>, R: DynSend, C: FromIterator<R>>(
    t: T,
    map: impl Fn(I) -> R + Dss,
) -> C {
    parallel_guard(|guard| {
        if let Some(proof) = mode::check_dyn_thread_safe() {
            let map = proof.derive(map);

            let mut items: Vec<(Option<I>, Option<R>)> =
                t.into_iter().map(|i| (Some(i), None)).collect();

            par_slice_mut(
                &mut items,
                guard,
                |i| {
                    i.1 = Some(map(i.0.take().unwrap()));
                },
                proof,
            );

            items.into_iter().filter_map(|i| i.1).collect()
        } else {
            t.into_iter().filter_map(|i| guard.run(|| map(i))).collect()
        }
    })
}

pub fn broadcast<R: DynSend>(op: impl Fn(usize) -> R + DynSync) -> Vec<R> {
    if let Some(proof) = mode::check_dyn_thread_safe() {
        let op = proof.derive(op);
        let results = rustc_thread_pool::broadcast(|context| op.derive(op(context.index())));
        results.into_iter().map(|r| r.into_inner()).collect()
    } else {
        vec![op(0)]
    }
}
