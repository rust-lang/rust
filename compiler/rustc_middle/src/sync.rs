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
        (impl $fblock:block [$($c:expr,)*] [$block:expr $(, $rest:expr)*]) => {
            parallel!(impl $fblock [$block, $($c,)*] [$($rest),*])
        };
        (impl $fblock:block [$($blocks:expr,)*] []) => {
            #[allow(unreachable_code)]
            let n = 1 $(+ 'a: { break 'a 1; let _ = || $blocks; })*;
            $crate::sync::parallel_guard(|guard| {
                $crate::sync::scope(n, |mut s| {
                    $(
                        let block = $crate::sync::FromDyn::from(|| $blocks);
                        s.spawn(move || {
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

// This function only works when `is_dyn_thread_safe()`.
pub fn scope<'scope, OP, R>(spawn_limit: u64, op: OP) -> R
where
    OP: for<'a, 'tcx> FnOnce(Scope<'a, 'scope>) -> R + DynSend,
    R: DynSend,
{
    let op = FromDyn::from(op);
    rustc_thread_pool::scope(|scope| {
        FromDyn::from(op.into_inner()(Scope { scope, next_branch: 0, branch_limit: spawn_limit }))
    })
    .into_inner()
}

pub struct Scope<'a, 'scope> {
    scope: &'a rustc_thread_pool::Scope<'scope>,
    branch_limit: u64,
    next_branch: u64,
}

impl<'a, 'scope> Scope<'a, 'scope> {
    pub fn spawn<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        if self.next_branch >= self.branch_limit {
            panic!("number of spawns exceeded the spawn_limit = {}", self.branch_limit);
        }
        let query_branch = self.next_branch;
        self.next_branch += 1;
        branch_context(query_branch, self.branch_limit, || self.scope.spawn(|_| f()));
    }
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
                query: Some(QueryInclusion {
                    id,
                    branch: branch.branch(branch_num, branch_space),
                }),
                ..*icx
            };
            tls::enter_context(&icx, f)
        } else {
            f()
        }
    })
}
