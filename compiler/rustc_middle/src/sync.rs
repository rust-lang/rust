use parking_lot::Mutex;
pub use rustc_data_structures::marker::{DynSend, DynSync};
pub use rustc_data_structures::sync::*;

use crate::query::QueryInclusion;
pub use crate::ty::tls;

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

/// Runs the functions in parallel.
///
/// The first function is executed immediately on the current thread.
/// Use that for the longest running function for better scheduling.
pub fn par_fns(funcs: &mut [&mut (dyn FnMut() + DynSend)]) {
    parallel_guard(|guard: &ParallelGuard| {
        if is_dyn_thread_safe() {
            let func_count = funcs.len().try_into().unwrap();
            let funcs = FromDyn::from(funcs);
            rustc_thread_pool::scope(|s| {
                let Some((first, rest)) = funcs.into_inner().split_at_mut_checked(1) else {
                    return;
                };

                // Reverse the order of the later functions since Rayon executes them in reverse
                // order when using a single thread. This ensures the execution order matches
                // that of a single threaded rustc.
                for (i, f) in rest.iter_mut().enumerate().rev() {
                    let f = FromDyn::from(f);
                    s.spawn(move |_| {
                        branch_context((i + 1).try_into().unwrap(), func_count, || {
                            guard.run(|| (f.into_inner())())
                        });
                    });
                }

                // Run the first function without spawning to
                // ensure it executes immediately on this thread.
                branch_context(0, func_count, || guard.run(|| first[0]()));
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
    if is_dyn_thread_safe() {
        let oper_a = FromDyn::from(oper_a);
        let oper_b = FromDyn::from(oper_b);
        let (a, b) = parallel_guard(|guard| {
            let task_a = move || guard.run(move || FromDyn::from(oper_a.into_inner()()));
            let task_b = move || guard.run(move || FromDyn::from(oper_b.into_inner()()));
            rustc_thread_pool::join(
                || branch_context(0, 2, task_a),
                || branch_context(1, 2, task_b),
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
    match items {
        [] => return,
        [item] => {
            guard.run(|| for_each(item));
            return;
        }
        _ => (),
    }

    let for_each = FromDyn::from(for_each);
    let mut items = for_each.derive(items);
    rustc_thread_pool::scope(|s| {
        let for_each = &for_each;
        let proof = items.derive(());

        const MAX_GROUP_COUNT: usize = 128;
        let group_size = items.len().div_ceil(MAX_GROUP_COUNT);
        let mut groups = items.chunks_mut(group_size).enumerate();
        let group_count = groups.len().try_into().unwrap();

        let Some((_, first_group)) = groups.next() else { return };

        // Reverse the order of the later functions since Rayon executes them in reverse
        // order when using a single thread. This ensures the execution order matches
        // that of a single threaded rustc.
        for (i, group) in groups.rev() {
            let group = proof.derive(group);
            s.spawn(move |_| {
                branch_context(i.try_into().unwrap(), group_count, || {
                    let mut group = group;
                    for i in group.iter_mut() {
                        guard.run(|| for_each(i));
                    }
                })
            });
        }

        // Run the first function without spawning to
        // ensure it executes immediately on this thread.
        branch_context(0, group_count, || {
            for i in first_group.iter_mut() {
                guard.run(|| for_each(i));
            }
        });
    });
}

pub fn par_for_each_in<I: DynSend, T: IntoIterator<Item = I>>(
    t: T,
    for_each: impl Fn(&I) + DynSync + DynSend,
) {
    parallel_guard(|guard| {
        if is_dyn_thread_safe() {
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
        if is_dyn_thread_safe() {
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
        if is_dyn_thread_safe() {
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

/// Append `i`-th branch out of `n` branches to `icx.query.branch` to track inside of
/// which parallel task every query call is performed.
///
/// See [`rustc_data_structures::tree_node_index::TreeNodeIndex`].
fn branch_context<F, R>(i: u64, n: u64, f: F) -> R
where
    F: FnOnce() -> R,
{
    tls::with_context_opt(|icx| {
        if let Some(icx) = icx
            && let Some(QueryInclusion { id, branch }) = icx.query
        {
            let icx = tls::ImplicitCtxt {
                query: Some(QueryInclusion { id, branch: branch.branch(i, n) }),
                ..*icx
            };
            tls::enter_context(&icx, f)
        } else {
            f()
        }
    })
}
