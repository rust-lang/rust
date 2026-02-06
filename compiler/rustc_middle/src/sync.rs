use parking_lot::Mutex;
pub use rustc_data_structures::marker::{DynSend, DynSync};
pub use rustc_data_structures::sync::*;

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
            let funcs = FromDyn::from(funcs);
            rustc_thread_pool::scope(|s| {
                let Some((first, rest)) = funcs.into_inner().split_at_mut_checked(1) else {
                    return;
                };

                // Reverse the order of the later functions since Rayon executes them in reverse
                // order when using a single thread. This ensures the execution order matches
                // that of a single threaded rustc.
                for f in rest.iter_mut().rev() {
                    let f = FromDyn::from(f);
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
    if is_dyn_thread_safe() {
        let oper_a = FromDyn::from(oper_a);
        let oper_b = FromDyn::from(oper_b);
        let (a, b) = parallel_guard(|guard| {
            rustc_thread_pool::join(
                move || guard.run(move || FromDyn::from(oper_a.into_inner()())),
                move || guard.run(move || FromDyn::from(oper_b.into_inner()())),
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
    let for_each = FromDyn::from(for_each);
    let mut items = for_each.derive(items);
    rustc_thread_pool::scope(|s| {
        let proof = items.derive(());
        let group_size = std::cmp::max(items.len() / 128, 1);
        for group in items.chunks_mut(group_size) {
            let group = proof.derive(group);
            s.spawn(|_| {
                let mut group = group;
                for i in group.iter_mut() {
                    guard.run(|| for_each(i));
                }
            });
        }
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
