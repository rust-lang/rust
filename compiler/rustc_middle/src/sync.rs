use parking_lot::Mutex;
pub use rustc_data_structures::marker::{DynSend, DynSync};
pub use rustc_data_structures::sync::*;
use rustc_query_system::query::QueryInclusion;

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

/// Runs a list of blocks in parallel. The first block is executed immediately on
/// the current thread. Use that for the longest running block.
#[macro_export]
macro_rules! parallel {
        ($($blocks:block),*) => {
            $crate::sync::parallel_guard(|guard| {
                if $crate::sync::is_dyn_thread_safe() {
                    let proof = $crate::sync::FromDyn::from(());
                    parallel!(impl proof,
                        $(({
                            let block = || $blocks;
                            move || { guard.run(block); }
                        })),*
                    );
                } else {
                    $(guard.run(|| $blocks);)*
                }
            });
        };
        (impl $proof:expr, $f1:expr, $f2:expr, $f3:expr, $f4:expr, $f5:expr, $f6:expr, $f7:expr, $f8:expr, $f9:expr, $f10:expr, $f11:expr, $f12:expr, $f13:expr, $f14:expr, $f15:expr, $f16:expr, $($rest:expr),+) => {
            std::compiler_error!("`parallel!` only supports up to 16 blocks")
        };
        (impl $proof:expr, $f1:expr, $f2:expr, $f3:expr, $f4:expr, $f5:expr, $f6:expr, $f7:expr, $f8:expr, $($rest:expr),+) => {
            $crate::sync::parallel_macro_internal_join(
                move || parallel!(impl $proof, $f1, $f2, $f3, $f4, $f5, $f6, $f7, $f8),
                move || parallel!(impl $proof, $($rest),+),
                $proof,
            );
        };
        (impl $proof:expr, $f1:expr, $f2:expr, $f3:expr, $f4:expr, $($rest:expr),+) => {
            $crate::sync::parallel_macro_internal_join(move || parallel!(impl $proof, $f1, $f2, $f3, $f4), move || parallel!(impl $proof, $($rest),+), $proof);
        };
        (impl $proof:expr, $f1:expr, $f2:expr, $($rest:expr),+) => {
            $crate::sync::parallel_macro_internal_join(move || parallel!(impl $proof, $f1, $f2), move || parallel!(impl $proof, $($rest),+), $proof);
        };
        (impl $proof:expr, $f1:expr, $f2:expr) => {
            $crate::sync::parallel_macro_internal_join($f1, $f2, $proof);
        };
        (impl $proof:expr, $f1:expr) => {
            ($f1)();
        };
    }

#[inline]
pub fn join<A, B, RA: DynSend, RB: DynSend>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + DynSend,
    B: FnOnce() -> RB + DynSend,
{
    if is_dyn_thread_safe() {
        let oper_a = FromDyn::from(oper_a);
        let oper_b = FromDyn::from(oper_b);
        let (a, b) = parallel_guard(|guard| {
            raw_branched_join(
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
    struct State<'a, F> {
        for_each: FromDyn<F>,
        guard: &'a ParallelGuard,
        group: usize,
    }

    fn par_rec<I: DynSend, F: Fn(&mut I) + DynSync + DynSend>(
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
            raw_branched_join(move || par_rec(*left, state), move || par_rec(*right, state));
        }
    }

    let state = State {
        for_each: FromDyn::from(for_each),
        guard,
        group: std::cmp::max(items.len() / 128, 1),
    };
    par_rec(items, &state)
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

/// Used internally in `parallel` macro. DO NOT USE DIRECTLY!
pub fn parallel_macro_internal_join<A, B>(oper_a: A, oper_b: B, proof: FromDyn<()>)
where
    A: FnOnce() + DynSend,
    B: FnOnce() + DynSend,
{
    let oper_a = proof.derive(oper_a);
    let oper_b = proof.derive(oper_b);
    raw_branched_join(move || oper_a.into_inner()(), move || oper_b.into_inner()());
}

fn raw_branched_join<A, B, RA: Send, RB: Send>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
{
    rustc_thread_pool::join(|| branch_context(0, 2, oper_a), || branch_context(1, 2, oper_b))
}

fn branch_context<F, R>(branch_num: u64, branch_space: u64, f: F) -> R
where
    F: FnOnce() -> R,
{
    tls::with_context_opt(|icx| {
        if let Some(icx) = icx
            && let Some(QueryInclusion { id, branch }) = icx.query
        {
            let icx = tls::ImplicitCtxt {
                query: Some(QueryInclusion { id, branch: branch.branch(branch_num, branch_space) }),
                ..*icx
            };
            tls::enter_context(&icx, f)
        } else {
            f()
        }
    })
}
