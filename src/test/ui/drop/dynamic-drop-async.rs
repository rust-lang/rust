// Test that values are not leaked in async functions, even in the cases where:
// * Dropping one of the values panics while running the future.
// * The future is dropped at one of its suspend points.
// * Dropping one of the values panics while dropping the future.

// run-pass
// edition:2018
// ignore-wasm32-bare compiled with panic=abort by default

#![allow(unused)]

use std::{
    cell::{Cell, RefCell},
    future::Future,
    marker::Unpin,
    panic,
    pin::Pin,
    ptr,
    rc::Rc,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    usize,
};

struct InjectedFailure;

struct Defer<T> {
    ready: bool,
    value: Option<T>,
}

impl<T: Unpin> Future for Defer<T> {
    type Output = T;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        if self.ready {
            Poll::Ready(self.value.take().unwrap())
        } else {
            self.ready = true;
            Poll::Pending
        }
    }
}

/// Allocator tracks the creation and destruction of `Ptr`s.
/// The `failing_op`-th operation will panic.
struct Allocator {
    data: RefCell<Vec<bool>>,
    failing_op: usize,
    cur_ops: Cell<usize>,
}

impl panic::UnwindSafe for Allocator {}
impl panic::RefUnwindSafe for Allocator {}

impl Drop for Allocator {
    fn drop(&mut self) {
        let data = self.data.borrow();
        if data.iter().any(|d| *d) {
            panic!("missing free: {:?}", data);
        }
    }
}

impl Allocator {
    fn new(failing_op: usize) -> Self {
        Allocator { failing_op, cur_ops: Cell::new(0), data: RefCell::new(vec![]) }
    }
    fn alloc(&self) -> impl Future<Output = Ptr<'_>> + '_ {
        self.fallible_operation();

        let mut data = self.data.borrow_mut();

        let addr = data.len();
        data.push(true);
        Defer { ready: false, value: Some(Ptr(addr, self)) }
    }
    fn fallible_operation(&self) {
        self.cur_ops.set(self.cur_ops.get() + 1);

        if self.cur_ops.get() == self.failing_op {
            panic!(InjectedFailure);
        }
    }
}

// Type that tracks whether it was dropped and can panic when it's created or
// destroyed.
struct Ptr<'a>(usize, &'a Allocator);
impl<'a> Drop for Ptr<'a> {
    fn drop(&mut self) {
        match self.1.data.borrow_mut()[self.0] {
            false => panic!("double free at index {:?}", self.0),
            ref mut d => *d = false,
        }

        self.1.fallible_operation();
    }
}

async fn dynamic_init(a: Rc<Allocator>, c: bool) {
    let _x;
    if c {
        _x = Some(a.alloc().await);
    }
}

async fn dynamic_drop(a: Rc<Allocator>, c: bool) {
    let x = a.alloc().await;
    if c {
        Some(x)
    } else {
        None
    };
}

struct TwoPtrs<'a>(Ptr<'a>, Ptr<'a>);
async fn struct_dynamic_drop(a: Rc<Allocator>, c0: bool, c1: bool, c: bool) {
    for i in 0..2 {
        let x;
        let y;
        if (c0 && i == 0) || (c1 && i == 1) {
            x = (a.alloc().await, a.alloc().await, a.alloc().await);
            y = TwoPtrs(a.alloc().await, a.alloc().await);
            if c {
                drop(x.1);
                a.alloc().await;
                drop(y.0);
                a.alloc().await;
            }
        }
    }
}

async fn field_assignment(a: Rc<Allocator>, c0: bool) {
    let mut x = (TwoPtrs(a.alloc().await, a.alloc().await), a.alloc().await);

    x.1 = a.alloc().await;
    x.1 = a.alloc().await;

    let f = (x.0).0;
    a.alloc().await;
    if c0 {
        (x.0).0 = f;
    }
    a.alloc().await;
}

async fn assignment(a: Rc<Allocator>, c0: bool, c1: bool) {
    let mut _v = a.alloc().await;
    let mut _w = a.alloc().await;
    if c0 {
        drop(_v);
    }
    _v = _w;
    if c1 {
        _w = a.alloc().await;
    }
}

async fn array_simple(a: Rc<Allocator>) {
    let _x = [a.alloc().await, a.alloc().await, a.alloc().await, a.alloc().await];
}

async fn vec_simple(a: Rc<Allocator>) {
    let _x = vec![a.alloc().await, a.alloc().await, a.alloc().await, a.alloc().await];
}

async fn mixed_drop_and_nondrop(a: Rc<Allocator>) {
    // check that destructor panics handle drop
    // and non-drop blocks in the same scope correctly.
    //
    // Surprisingly enough, this used to not work.
    let (x, y, z);
    x = a.alloc().await;
    y = 5;
    z = a.alloc().await;
}

#[allow(unreachable_code)]
async fn vec_unreachable(a: Rc<Allocator>) {
    let _x = vec![a.alloc().await, a.alloc().await, a.alloc().await, return];
}

async fn slice_pattern_one_of(a: Rc<Allocator>, i: usize) {
    let array = [a.alloc().await, a.alloc().await, a.alloc().await, a.alloc().await];
    let _x = match i {
        0 => {
            let [a, ..] = array;
            a
        }
        1 => {
            let [_, a, ..] = array;
            a
        }
        2 => {
            let [_, _, a, _] = array;
            a
        }
        3 => {
            let [_, _, _, a] = array;
            a
        }
        _ => panic!("unmatched"),
    };
    a.alloc().await;
}

async fn subslice_pattern_from_end_with_drop(a: Rc<Allocator>, arg: bool, arg2: bool) {
    let arr = [a.alloc().await, a.alloc().await, a.alloc().await, a.alloc().await, a.alloc().await];
    if arg2 {
        drop(arr);
        return;
    }

    if arg {
        let [.., _x, _] = arr;
    } else {
        let [_, _y @ ..] = arr;
    }
    a.alloc().await;
}

async fn subslice_pattern_reassign(a: Rc<Allocator>) {
    let mut ar = [a.alloc().await, a.alloc().await, a.alloc().await];
    let [_, _, _x] = ar;
    ar = [a.alloc().await, a.alloc().await, a.alloc().await];
    let [_, _y @ ..] = ar;
    a.alloc().await;
}

async fn move_ref_pattern(a: Rc<Allocator>) {
    let mut tup = (a.alloc().await, a.alloc().await, a.alloc().await, a.alloc().await);
    let (ref _a, ref mut _b, _c, mut _d) = tup;
    a.alloc().await;
}

fn run_test<F, G>(cx: &mut Context<'_>, ref f: F)
where
    F: Fn(Rc<Allocator>) -> G,
    G: Future<Output = ()>,
{
    for polls in 0.. {
        // Run without any panics to find which operations happen after the
        // penultimate `poll`.
        let first_alloc = Rc::new(Allocator::new(usize::MAX));
        let mut fut = Box::pin(f(first_alloc.clone()));
        let mut ops_before_last_poll = 0;
        let mut completed = false;
        for _ in 0..polls {
            ops_before_last_poll = first_alloc.cur_ops.get();
            if let Poll::Ready(()) = fut.as_mut().poll(cx) {
                completed = true;
            }
        }
        drop(fut);

        // Start at `ops_before_last_poll` so that we will always be able to
        // `poll` the expected number of times.
        for failing_op in ops_before_last_poll..first_alloc.cur_ops.get() {
            let alloc = Rc::new(Allocator::new(failing_op + 1));
            let f = &f;
            let cx = &mut *cx;
            let result = panic::catch_unwind(panic::AssertUnwindSafe(move || {
                let mut fut = Box::pin(f(alloc));
                for _ in 0..polls {
                    let _ = fut.as_mut().poll(cx);
                }
                drop(fut);
            }));
            match result {
                Ok(..) => panic!("test executed more ops on first call"),
                Err(e) => {
                    if e.downcast_ref::<InjectedFailure>().is_none() {
                        panic::resume_unwind(e);
                    }
                }
            }
        }

        if completed {
            break;
        }
    }
}

fn clone_waker(data: *const ()) -> RawWaker {
    RawWaker::new(data, &RawWakerVTable::new(clone_waker, drop, drop, drop))
}

fn main() {
    let waker = unsafe { Waker::from_raw(clone_waker(ptr::null())) };
    let context = &mut Context::from_waker(&waker);

    run_test(context, |a| dynamic_init(a, false));
    run_test(context, |a| dynamic_init(a, true));
    run_test(context, |a| dynamic_drop(a, false));
    run_test(context, |a| dynamic_drop(a, true));

    run_test(context, |a| assignment(a, false, false));
    run_test(context, |a| assignment(a, false, true));
    run_test(context, |a| assignment(a, true, false));
    run_test(context, |a| assignment(a, true, true));

    run_test(context, |a| array_simple(a));
    run_test(context, |a| vec_simple(a));
    run_test(context, |a| vec_unreachable(a));

    run_test(context, |a| struct_dynamic_drop(a, false, false, false));
    run_test(context, |a| struct_dynamic_drop(a, false, false, true));
    run_test(context, |a| struct_dynamic_drop(a, false, true, false));
    run_test(context, |a| struct_dynamic_drop(a, false, true, true));
    run_test(context, |a| struct_dynamic_drop(a, true, false, false));
    run_test(context, |a| struct_dynamic_drop(a, true, false, true));
    run_test(context, |a| struct_dynamic_drop(a, true, true, false));
    run_test(context, |a| struct_dynamic_drop(a, true, true, true));

    run_test(context, |a| field_assignment(a, false));
    run_test(context, |a| field_assignment(a, true));

    run_test(context, |a| mixed_drop_and_nondrop(a));

    run_test(context, |a| slice_pattern_one_of(a, 0));
    run_test(context, |a| slice_pattern_one_of(a, 1));
    run_test(context, |a| slice_pattern_one_of(a, 2));
    run_test(context, |a| slice_pattern_one_of(a, 3));

    run_test(context, |a| subslice_pattern_from_end_with_drop(a, true, true));
    run_test(context, |a| subslice_pattern_from_end_with_drop(a, true, false));
    run_test(context, |a| subslice_pattern_from_end_with_drop(a, false, true));
    run_test(context, |a| subslice_pattern_from_end_with_drop(a, false, false));
    run_test(context, |a| subslice_pattern_reassign(a));

    run_test(context, |a| move_ref_pattern(a));
}
