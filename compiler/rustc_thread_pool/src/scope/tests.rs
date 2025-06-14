use std::iter::once;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::vec;

use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::{Scope, ScopeFifo, ThreadPoolBuilder, scope, scope_fifo, unwind};

#[test]
fn scope_empty() {
    scope(|_| {});
}

#[test]
fn scope_result() {
    let x = scope(|_| 22);
    assert_eq!(x, 22);
}

#[test]
fn scope_two() {
    let counter = &AtomicUsize::new(0);
    scope(|s| {
        s.spawn(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        s.spawn(move |_| {
            counter.fetch_add(10, Ordering::SeqCst);
        });
    });

    let v = counter.load(Ordering::SeqCst);
    assert_eq!(v, 11);
}

#[test]
fn scope_divide_and_conquer() {
    let counter_p = &AtomicUsize::new(0);
    scope(|s| s.spawn(move |s| divide_and_conquer(s, counter_p, 1024)));

    let counter_s = &AtomicUsize::new(0);
    divide_and_conquer_seq(counter_s, 1024);

    let p = counter_p.load(Ordering::SeqCst);
    let s = counter_s.load(Ordering::SeqCst);
    assert_eq!(p, s);
}

fn divide_and_conquer<'scope>(scope: &Scope<'scope>, counter: &'scope AtomicUsize, size: usize) {
    if size > 1 {
        scope.spawn(move |scope| divide_and_conquer(scope, counter, size / 2));
        scope.spawn(move |scope| divide_and_conquer(scope, counter, size / 2));
    } else {
        // count the leaves
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

fn divide_and_conquer_seq(counter: &AtomicUsize, size: usize) {
    if size > 1 {
        divide_and_conquer_seq(counter, size / 2);
        divide_and_conquer_seq(counter, size / 2);
    } else {
        // count the leaves
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

struct Tree<T: Send> {
    value: T,
    children: Vec<Tree<T>>,
}

impl<T: Send> Tree<T> {
    fn iter(&self) -> vec::IntoIter<&T> {
        once(&self.value)
            .chain(self.children.iter().flat_map(Tree::iter))
            .collect::<Vec<_>>() // seems like it shouldn't be needed... but prevents overflow
            .into_iter()
    }

    fn update<OP>(&mut self, op: OP)
    where
        OP: Fn(&mut T) + Sync,
        T: Send,
    {
        scope(|s| self.update_in_scope(&op, s));
    }

    fn update_in_scope<'scope, OP>(&'scope mut self, op: &'scope OP, scope: &Scope<'scope>)
    where
        OP: Fn(&mut T) + Sync,
    {
        let Tree { ref mut value, ref mut children } = *self;
        scope.spawn(move |scope| {
            for child in children {
                scope.spawn(move |scope| child.update_in_scope(op, scope));
            }
        });

        op(value);
    }
}

fn random_tree(depth: usize) -> Tree<u32> {
    assert!(depth > 0);
    let mut seed = <XorShiftRng as SeedableRng>::Seed::default();
    (0..).zip(seed.as_mut()).for_each(|(i, x)| *x = i);
    let mut rng = XorShiftRng::from_seed(seed);
    random_tree1(depth, &mut rng)
}

fn random_tree1(depth: usize, rng: &mut XorShiftRng) -> Tree<u32> {
    let children = if depth == 0 {
        vec![]
    } else {
        (0..rng.random_range(0..4)) // somewhere between 0 and 3 children at each level
            .map(|_| random_tree1(depth - 1, rng))
            .collect()
    };

    Tree { value: rng.random_range(0..1_000_000), children }
}

#[test]
fn update_tree() {
    let mut tree: Tree<u32> = random_tree(10);
    let values: Vec<u32> = tree.iter().cloned().collect();
    tree.update(|v| *v += 1);
    let new_values: Vec<u32> = tree.iter().cloned().collect();
    assert_eq!(values.len(), new_values.len());
    for (&i, &j) in values.iter().zip(&new_values) {
        assert_eq!(i + 1, j);
    }
}

/// Check that if you have a chain of scoped tasks where T0 spawns T1
/// spawns T2 and so forth down to Tn, the stack space should not grow
/// linearly with N. We test this by some unsafe hackery and
/// permitting an approx 10% change with a 10x input change.
#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn linear_stack_growth() {
    let builder = ThreadPoolBuilder::new().num_threads(1);
    let pool = builder.build().unwrap();
    pool.install(|| {
        let mut max_diff = Mutex::new(0);
        let bottom_of_stack = 0;
        scope(|s| the_final_countdown(s, &bottom_of_stack, &max_diff, 5));
        let diff_when_5 = *max_diff.get_mut().unwrap() as f64;

        scope(|s| the_final_countdown(s, &bottom_of_stack, &max_diff, 500));
        let diff_when_500 = *max_diff.get_mut().unwrap() as f64;

        let ratio = diff_when_5 / diff_when_500;
        assert!(ratio > 0.9 && ratio < 1.1, "stack usage ratio out of bounds: {}", ratio);
    });
}

fn the_final_countdown<'scope>(
    s: &Scope<'scope>,
    bottom_of_stack: &'scope i32,
    max: &'scope Mutex<usize>,
    n: usize,
) {
    let top_of_stack = 0;
    let p = bottom_of_stack as *const i32 as usize;
    let q = &top_of_stack as *const i32 as usize;
    let diff = if p > q { p - q } else { q - p };

    let mut data = max.lock().unwrap();
    *data = Ord::max(diff, *data);

    if n > 0 {
        s.spawn(move |s| the_final_countdown(s, bottom_of_stack, max, n - 1));
    }
}

#[test]
#[should_panic(expected = "Hello, world!")]
fn panic_propagate_scope() {
    scope(|_| panic!("Hello, world!"));
}

#[test]
#[should_panic(expected = "Hello, world!")]
fn panic_propagate_spawn() {
    scope(|s| s.spawn(|_| panic!("Hello, world!")));
}

#[test]
#[should_panic(expected = "Hello, world!")]
fn panic_propagate_nested_spawn() {
    scope(|s| s.spawn(|s| s.spawn(|s| s.spawn(|_| panic!("Hello, world!")))));
}

#[test]
#[should_panic(expected = "Hello, world!")]
fn panic_propagate_nested_scope_spawn() {
    scope(|s| s.spawn(|_| scope(|s| s.spawn(|_| panic!("Hello, world!")))));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn panic_propagate_still_execute_1() {
    let mut x = false;
    let result = unwind::halt_unwinding(|| {
        scope(|s| {
            s.spawn(|_| panic!("Hello, world!")); // job A
            s.spawn(|_| x = true); // job B, should still execute even though A panics
        });
    });
    match result {
        Ok(_) => panic!("failed to propagate panic"),
        Err(_) => assert!(x, "job b failed to execute"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn panic_propagate_still_execute_2() {
    let mut x = false;
    let result = unwind::halt_unwinding(|| {
        scope(|s| {
            s.spawn(|_| x = true); // job B, should still execute even though A panics
            s.spawn(|_| panic!("Hello, world!")); // job A
        });
    });
    match result {
        Ok(_) => panic!("failed to propagate panic"),
        Err(_) => assert!(x, "job b failed to execute"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn panic_propagate_still_execute_3() {
    let mut x = false;
    let result = unwind::halt_unwinding(|| {
        scope(|s| {
            s.spawn(|_| x = true); // spawned job should still execute despite later panic
            panic!("Hello, world!");
        });
    });
    match result {
        Ok(_) => panic!("failed to propagate panic"),
        Err(_) => assert!(x, "panic after spawn, spawn failed to execute"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn panic_propagate_still_execute_4() {
    let mut x = false;
    let result = unwind::halt_unwinding(|| {
        scope(|s| {
            s.spawn(|_| panic!("Hello, world!"));
            x = true;
        });
    });
    match result {
        Ok(_) => panic!("failed to propagate panic"),
        Err(_) => assert!(x, "panic in spawn tainted scope"),
    }
}

macro_rules! test_order {
    ($scope:ident => $spawn:ident) => {{
        let builder = ThreadPoolBuilder::new().num_threads(1);
        let pool = builder.build().unwrap();
        pool.install(|| {
            let vec = Mutex::new(vec![]);
            $scope(|scope| {
                let vec = &vec;
                for i in 0..10 {
                    scope.$spawn(move |scope| {
                        for j in 0..10 {
                            scope.$spawn(move |_| {
                                vec.lock().unwrap().push(i * 10 + j);
                            });
                        }
                    });
                }
            });
            vec.into_inner().unwrap()
        })
    }};
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn lifo_order() {
    // In the absence of stealing, `scope()` runs its `spawn()` jobs in LIFO order.
    let vec = test_order!(scope => spawn);
    let expected: Vec<i32> = (0..100).rev().collect(); // LIFO -> reversed
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn fifo_order() {
    // In the absence of stealing, `scope_fifo()` runs its `spawn_fifo()` jobs in FIFO order.
    let vec = test_order!(scope_fifo => spawn_fifo);
    let expected: Vec<i32> = (0..100).collect(); // FIFO -> natural order
    assert_eq!(vec, expected);
}

macro_rules! test_nested_order {
    ($outer_scope:ident => $outer_spawn:ident,
     $inner_scope:ident => $inner_spawn:ident) => {{
        let builder = ThreadPoolBuilder::new().num_threads(1);
        let pool = builder.build().unwrap();
        pool.install(|| {
            let vec = Mutex::new(vec![]);
            $outer_scope(|scope| {
                let vec = &vec;
                for i in 0..10 {
                    scope.$outer_spawn(move |_| {
                        $inner_scope(|scope| {
                            for j in 0..10 {
                                scope.$inner_spawn(move |_| {
                                    vec.lock().unwrap().push(i * 10 + j);
                                });
                            }
                        });
                    });
                }
            });
            vec.into_inner().unwrap()
        })
    }};
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn nested_lifo_order() {
    // In the absence of stealing, `scope()` runs its `spawn()` jobs in LIFO order.
    let vec = test_nested_order!(scope => spawn, scope => spawn);
    let expected: Vec<i32> = (0..100).rev().collect(); // LIFO -> reversed
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn nested_fifo_order() {
    // In the absence of stealing, `scope_fifo()` runs its `spawn_fifo()` jobs in FIFO order.
    let vec = test_nested_order!(scope_fifo => spawn_fifo, scope_fifo => spawn_fifo);
    let expected: Vec<i32> = (0..100).collect(); // FIFO -> natural order
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn nested_lifo_fifo_order() {
    // LIFO on the outside, FIFO on the inside
    let vec = test_nested_order!(scope => spawn, scope_fifo => spawn_fifo);
    let expected: Vec<i32> = (0..10).rev().flat_map(|i| (0..10).map(move |j| i * 10 + j)).collect();
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn nested_fifo_lifo_order() {
    // FIFO on the outside, LIFO on the inside
    let vec = test_nested_order!(scope_fifo => spawn_fifo, scope => spawn);
    let expected: Vec<i32> = (0..10).flat_map(|i| (0..10).rev().map(move |j| i * 10 + j)).collect();
    assert_eq!(vec, expected);
}

macro_rules! spawn_push {
    ($scope:ident . $spawn:ident, $vec:ident, $i:expr) => {{
        $scope.$spawn(move |_| $vec.lock().unwrap().push($i));
    }};
}

/// Test spawns pushing a series of numbers, interleaved
/// such that negative values are using an inner scope.
macro_rules! test_mixed_order {
    ($outer_scope:ident => $outer_spawn:ident,
     $inner_scope:ident => $inner_spawn:ident) => {{
        let builder = ThreadPoolBuilder::new().num_threads(1);
        let pool = builder.build().unwrap();
        pool.install(|| {
            let vec = Mutex::new(vec![]);
            $outer_scope(|outer_scope| {
                let vec = &vec;
                spawn_push!(outer_scope.$outer_spawn, vec, 0);
                $inner_scope(|inner_scope| {
                    spawn_push!(inner_scope.$inner_spawn, vec, -1);
                    spawn_push!(outer_scope.$outer_spawn, vec, 1);
                    spawn_push!(inner_scope.$inner_spawn, vec, -2);
                    spawn_push!(outer_scope.$outer_spawn, vec, 2);
                    spawn_push!(inner_scope.$inner_spawn, vec, -3);
                });
                spawn_push!(outer_scope.$outer_spawn, vec, 3);
            });
            vec.into_inner().unwrap()
        })
    }};
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn mixed_lifo_order() {
    // NB: the end of the inner scope makes us execute some of the outer scope
    // before they've all been spawned, so they're not perfectly LIFO.
    let vec = test_mixed_order!(scope => spawn, scope => spawn);
    let expected = vec![-3, 2, -2, 1, -1, 3, 0];
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn mixed_fifo_order() {
    let vec = test_mixed_order!(scope_fifo => spawn_fifo, scope_fifo => spawn_fifo);
    let expected = vec![-1, 0, -2, 1, -3, 2, 3];
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn mixed_lifo_fifo_order() {
    // NB: the end of the inner scope makes us execute some of the outer scope
    // before they've all been spawned, so they're not perfectly LIFO.
    let vec = test_mixed_order!(scope => spawn, scope_fifo => spawn_fifo);
    let expected = vec![-1, 2, -2, 1, -3, 3, 0];
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn mixed_fifo_lifo_order() {
    let vec = test_mixed_order!(scope_fifo => spawn_fifo, scope => spawn);
    let expected = vec![-3, 0, -2, 1, -1, 2, 3];
    assert_eq!(vec, expected);
}

#[test]
fn static_scope() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let mut range = 0..100;
    let sum = range.clone().sum();
    let iter = &mut range;

    COUNTER.store(0, Ordering::Relaxed);
    scope(|s: &Scope<'static>| {
        // While we're allowed the locally borrowed iterator,
        // the spawns must be static.
        for i in iter {
            s.spawn(move |_| {
                COUNTER.fetch_add(i, Ordering::Relaxed);
            });
        }
    });

    assert_eq!(COUNTER.load(Ordering::Relaxed), sum);
}

#[test]
fn static_scope_fifo() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let mut range = 0..100;
    let sum = range.clone().sum();
    let iter = &mut range;

    COUNTER.store(0, Ordering::Relaxed);
    scope_fifo(|s: &ScopeFifo<'static>| {
        // While we're allowed the locally borrowed iterator,
        // the spawns must be static.
        for i in iter {
            s.spawn_fifo(move |_| {
                COUNTER.fetch_add(i, Ordering::Relaxed);
            });
        }
    });

    assert_eq!(COUNTER.load(Ordering::Relaxed), sum);
}

#[test]
fn mixed_lifetime_scope() {
    fn increment<'slice, 'counter>(counters: &'slice [&'counter AtomicUsize]) {
        scope(move |s: &Scope<'counter>| {
            // We can borrow 'slice here, but the spawns can only borrow 'counter.
            for &c in counters {
                s.spawn(move |_| {
                    c.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
    }

    let counter = AtomicUsize::new(0);
    increment(&[&counter; 100]);
    assert_eq!(counter.into_inner(), 100);
}

#[test]
fn mixed_lifetime_scope_fifo() {
    fn increment<'slice, 'counter>(counters: &'slice [&'counter AtomicUsize]) {
        scope_fifo(move |s: &ScopeFifo<'counter>| {
            // We can borrow 'slice here, but the spawns can only borrow 'counter.
            for &c in counters {
                s.spawn_fifo(move |_| {
                    c.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
    }

    let counter = AtomicUsize::new(0);
    increment(&[&counter; 100]);
    assert_eq!(counter.into_inner(), 100);
}

#[test]
fn scope_spawn_broadcast() {
    let sum = AtomicUsize::new(0);
    let n = scope(|s| {
        s.spawn_broadcast(|_, ctx| {
            sum.fetch_add(ctx.index(), Ordering::Relaxed);
        });
        crate::current_num_threads()
    });
    assert_eq!(sum.into_inner(), n * (n - 1) / 2);
}

#[test]
fn scope_fifo_spawn_broadcast() {
    let sum = AtomicUsize::new(0);
    let n = scope_fifo(|s| {
        s.spawn_broadcast(|_, ctx| {
            sum.fetch_add(ctx.index(), Ordering::Relaxed);
        });
        crate::current_num_threads()
    });
    assert_eq!(sum.into_inner(), n * (n - 1) / 2);
}

#[test]
fn scope_spawn_broadcast_nested() {
    let sum = AtomicUsize::new(0);
    let n = scope(|s| {
        s.spawn_broadcast(|s, _| {
            s.spawn_broadcast(|_, ctx| {
                sum.fetch_add(ctx.index(), Ordering::Relaxed);
            });
        });
        crate::current_num_threads()
    });
    assert_eq!(sum.into_inner(), n * n * (n - 1) / 2);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn scope_spawn_broadcast_barrier() {
    let barrier = Barrier::new(8);
    let pool = ThreadPoolBuilder::new().num_threads(7).build().unwrap();
    pool.in_place_scope(|s| {
        s.spawn_broadcast(|_, _| {
            barrier.wait();
        });
        barrier.wait();
    });
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn scope_spawn_broadcast_panic_one() {
    let count = AtomicUsize::new(0);
    let pool = ThreadPoolBuilder::new().num_threads(7).build().unwrap();
    let result = crate::unwind::halt_unwinding(|| {
        pool.scope(|s| {
            s.spawn_broadcast(|_, ctx| {
                count.fetch_add(1, Ordering::Relaxed);
                if ctx.index() == 3 {
                    panic!("Hello, world!");
                }
            });
        });
    });
    assert_eq!(count.into_inner(), 7);
    assert!(result.is_err(), "broadcast panic should propagate!");
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn scope_spawn_broadcast_panic_many() {
    let count = AtomicUsize::new(0);
    let pool = ThreadPoolBuilder::new().num_threads(7).build().unwrap();
    let result = crate::unwind::halt_unwinding(|| {
        pool.scope(|s| {
            s.spawn_broadcast(|_, ctx| {
                count.fetch_add(1, Ordering::Relaxed);
                if ctx.index() % 2 == 0 {
                    panic!("Hello, world!");
                }
            });
        });
    });
    assert_eq!(count.into_inner(), 7);
    assert!(result.is_err(), "broadcast panic should propagate!");
}
