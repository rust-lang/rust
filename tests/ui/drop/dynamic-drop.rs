//@ run-pass
//@ needs-unwind

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
#![feature(if_let_guard)]

#![allow(unused_assignments)]
#![allow(unused_variables)]

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ops::Coroutine;
use std::panic;
use std::pin::Pin;

struct InjectedFailure;

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
        Allocator {
            failing_op: failing_op,
            cur_ops: Cell::new(0),
            data: RefCell::new(vec![])
        }
    }
    fn alloc(&self) -> Ptr<'_> {
        self.cur_ops.set(self.cur_ops.get() + 1);

        if self.cur_ops.get() == self.failing_op {
            panic::panic_any(InjectedFailure);
        }

        let mut data = self.data.borrow_mut();
        let addr = data.len();
        data.push(true);
        Ptr(addr, self)
    }
    // FIXME(#47949) Any use of this indicates a bug in rustc: we should never
    // be leaking values in the cases here.
    //
    // Creates a `Ptr<'_>` and checks that the allocated value is leaked if the
    // `failing_op` is in the list of exception.
    fn alloc_leaked(&self, exceptions: Vec<usize>) -> Ptr<'_> {
        let ptr = self.alloc();

        if exceptions.iter().any(|operation| *operation == self.failing_op) {
            let mut data = self.data.borrow_mut();
            data[ptr.0] = false;
        }
        ptr
    }
}

struct Ptr<'a>(usize, &'a Allocator);
impl<'a> Drop for Ptr<'a> {
    fn drop(&mut self) {
        match self.1.data.borrow_mut()[self.0] {
            false => {
                panic!("double free at index {:?}", self.0)
            }
            ref mut d => *d = false
        }

        self.1.cur_ops.set(self.1.cur_ops.get()+1);

        if self.1.cur_ops.get() == self.1.failing_op {
            panic::panic_any(InjectedFailure);
        }
    }
}

fn dynamic_init(a: &Allocator, c: bool) {
    let _x;
    if c {
        _x = Some(a.alloc());
    }
}

fn dynamic_drop(a: &Allocator, c: bool) {
    let x = a.alloc();
    if c {
        Some(x)
    } else {
        None
    };
}

struct TwoPtrs<'a>(Ptr<'a>, #[allow(dead_code)] Ptr<'a>);
fn struct_dynamic_drop(a: &Allocator, c0: bool, c1: bool, c: bool) {
    for i in 0..2 {
        let x;
        let y;
        if (c0 && i == 0) || (c1 && i == 1) {
            x = (a.alloc(), a.alloc(), a.alloc());
            y = TwoPtrs(a.alloc(), a.alloc());
            if c {
                drop(x.1);
                drop(y.0);
            }
        }
    }
}

fn field_assignment(a: &Allocator, c0: bool) {
    let mut x = (TwoPtrs(a.alloc(), a.alloc()), a.alloc());

    x.1 = a.alloc();
    x.1 = a.alloc();

    let f = (x.0).0;
    if c0 {
        (x.0).0 = f;
    }
}

fn assignment2(a: &Allocator, c0: bool, c1: bool) {
    let mut _v = a.alloc();
    let mut _w = a.alloc();
    if c0 {
        drop(_v);
    }
    _v = _w;
    if c1 {
        _w = a.alloc();
    }
}

fn assignment1(a: &Allocator, c0: bool) {
    let mut _v = a.alloc();
    let mut _w = a.alloc();
    if c0 {
        drop(_v);
    }
    _v = _w;
}

union Boxy<T> {
    a: ManuallyDrop<T>,
    b: ManuallyDrop<T>,
}

fn union1(a: &Allocator) {
    unsafe {
        let mut u = Boxy { a: ManuallyDrop::new(a.alloc()) };
        *u.b = a.alloc(); // drops first alloc
        drop(ManuallyDrop::into_inner(u.a));
    }
}

fn array_simple(a: &Allocator) {
    let _x = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
}

fn vec_simple(a: &Allocator) {
    let _x = vec![a.alloc(), a.alloc(), a.alloc(), a.alloc()];
}

fn coroutine(a: &Allocator, run_count: usize) {
    assert!(run_count < 4);

    let mut gen = #[coroutine] || {
        (a.alloc(),
         yield a.alloc(),
         a.alloc(),
         yield a.alloc()
         );
    };
    for _ in 0..run_count {
        Pin::new(&mut gen).resume(());
    }
}

fn mixed_drop_and_nondrop(a: &Allocator) {
    // check that destructor panics handle drop
    // and non-drop blocks in the same scope correctly.
    //
    // Surprisingly enough, this used to not work.
    let (x, y, z);
    x = a.alloc();
    y = 5;
    z = a.alloc();
}

#[allow(unreachable_code)]
fn vec_unreachable(a: &Allocator) {
    let _x = vec![a.alloc(), a.alloc(), a.alloc(), return];
}

fn slice_pattern_first(a: &Allocator) {
    let[_x, ..] = [a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_middle(a: &Allocator) {
    let[_, _x, _] = [a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_two(a: &Allocator) {
    let[_x, _, _y, _] = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_last(a: &Allocator) {
    let[.., _y] = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_one_of(a: &Allocator, i: usize) {
    let array = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
    let _x = match i {
        0 => { let [a, ..] = array; a }
        1 => { let [_, a, ..] = array; a }
        2 => { let [_, _, a, _] = array; a }
        3 => { let [_, _, _, a] = array; a }
        _ => panic!("unmatched"),
    };
}

fn subslice_pattern_from_end(a: &Allocator, arg: bool) {
    let a = [a.alloc(), a.alloc(), a.alloc()];
    if arg {
        let[.., _x, _] = a;
    } else {
        let[_, _y @ ..] = a;
    }
}

fn subslice_pattern_from_end_with_drop(a: &Allocator, arg: bool, arg2: bool) {
    let a = [a.alloc(), a.alloc(), a.alloc(), a.alloc(), a.alloc()];
    if arg2 {
        drop(a);
        return;
    }

    if arg {
        let[.., _x, _] = a;
    } else {
        let[_, _y @ ..] = a;
    }
}

fn slice_pattern_reassign(a: &Allocator) {
    let mut ar = [a.alloc(), a.alloc()];
    let[_, _x] = ar;
    ar = [a.alloc(), a.alloc()];
    let[.., _y] = ar;
}

fn subslice_pattern_reassign(a: &Allocator) {
    let mut ar = [a.alloc(), a.alloc(), a.alloc()];
    let[_, _, _x] = ar;
    ar = [a.alloc(), a.alloc(), a.alloc()];
    let[_, _y @ ..] = ar;
}

fn index_field_mixed_ends(a: &Allocator) {
    let ar = [(a.alloc(), a.alloc()), (a.alloc(), a.alloc())];
    let[(_x, _), ..] = ar;
    let[(_, _y), _] = ar;
    let[_, (_, _w)] = ar;
    let[.., (_z, _)] = ar;
}

fn subslice_mixed_min_lengths(a: &Allocator, c: i32) {
    let ar = [(a.alloc(), a.alloc()), (a.alloc(), a.alloc())];
    match c {
        0 => { let[_x, ..] = ar; }
        1 => { let[_x, _, ..] = ar; }
        2 => { let[_x, _] = ar; }
        3 => { let[(_x, _), _, ..] = ar; }
        4 => { let[.., (_x, _)] = ar; }
        5 => { let[.., (_x, _), _] = ar; }
        6 => { let [_y @ ..] = ar; }
        _ => { let [_y @ .., _] = ar; }
    }
}

fn bindings_after_at_dynamic_init_move(a: &Allocator, c: bool) {
    let foo = if c { Some(a.alloc()) } else { None };
    let _x;

    if let bar @ Some(_) = foo {
        _x = bar;
    }
}

fn bindings_after_at_dynamic_init_ref(a: &Allocator, c: bool) {
    let foo = if c { Some(a.alloc()) } else { None };
    let _x;

    if let bar @ Some(_baz) = &foo {
        _x = bar;
    }
}

fn bindings_after_at_dynamic_drop_move(a: &Allocator, c: bool) {
    let foo = if c { Some(a.alloc()) } else { None };

    if let bar @ Some(_) = foo {
        bar
    } else {
        None
    };
}

fn bindings_after_at_dynamic_drop_ref(a: &Allocator, c: bool) {
    let foo = if c { Some(a.alloc()) } else { None };

    if let bar @ Some(_baz) = &foo {
        bar
    } else {
        &None
    };
}

fn move_ref_pattern(a: &Allocator) {
    let mut tup = (a.alloc(), a.alloc(), a.alloc(), a.alloc());
    let (ref _a, ref mut _b, _c, mut _d) = tup;
}

fn if_let_guard(a: &Allocator, c: bool, d: i32) {
    let foo = if c { Some(a.alloc()) } else { None };

    match d == 0 {
        false if let Some(a) = foo => { let b = a; }
        true if let true = { drop(foo.unwrap_or_else(|| a.alloc())); d == 1 } => {}
        _ => {}
    }
}

fn if_let_guard_2(a: &Allocator, num: i32) {
    let d = a.alloc();
    match num {
        #[allow(irrefutable_let_patterns)]
        1 | 2 if let Ptr(ref _idx, _) = a.alloc() => {
            a.alloc();
        }
        _ => {}
    }
}

fn panic_after_return(a: &Allocator) -> Ptr<'_> {
    // Panic in the drop of `p` or `q` can leak
    let exceptions = vec![8, 9];
    a.alloc();
    let p = a.alloc();
    {
        a.alloc();
        let p = a.alloc();
        // FIXME (#47949) We leak values when we panic in a destructor after
        // evaluating an expression with `rustc_mir::build::Builder::into`.
        a.alloc_leaked(exceptions)
    }
}

fn panic_after_return_expr(a: &Allocator) -> Ptr<'_> {
    // Panic in the drop of `p` or `q` can leak
    let exceptions = vec![8, 9];
    a.alloc();
    let p = a.alloc();
    {
        a.alloc();
        let q = a.alloc();
        // FIXME (#47949)
        return a.alloc_leaked(exceptions);
    }
}

fn panic_after_init(a: &Allocator) {
    // Panic in the drop of `r` can leak
    let exceptions = vec![8];
    a.alloc();
    let p = a.alloc();
    let q = {
        a.alloc();
        let r = a.alloc();
        // FIXME (#47949)
        a.alloc_leaked(exceptions)
    };
}

fn panic_after_init_temp(a: &Allocator) {
    // Panic in the drop of `r` can leak
    let exceptions = vec![8];
    a.alloc();
    let p = a.alloc();
    {
        a.alloc();
        let r = a.alloc();
        // FIXME (#47949)
        a.alloc_leaked(exceptions)
    };
}

fn panic_after_init_by_loop(a: &Allocator) {
    // Panic in the drop of `r` can leak
    let exceptions = vec![8];
    a.alloc();
    let p = a.alloc();
    let q = loop {
        a.alloc();
        let r = a.alloc();
        // FIXME (#47949)
        break a.alloc_leaked(exceptions);
    };
}

fn run_test<F>(mut f: F)
    where F: FnMut(&Allocator)
{
    let first_alloc = Allocator::new(usize::MAX);
    f(&first_alloc);

    for failing_op in 1..first_alloc.cur_ops.get()+1 {
        let alloc = Allocator::new(failing_op);
        let alloc = &alloc;
        let f = panic::AssertUnwindSafe(&mut f);
        let result = panic::catch_unwind(move || {
            f.0(alloc);
        });
        match result {
            Ok(..) => panic!("test executed {} ops but now {}",
                             first_alloc.cur_ops.get(), alloc.cur_ops.get()),
            Err(e) => {
                if e.downcast_ref::<InjectedFailure>().is_none() {
                    panic::resume_unwind(e);
                }
            }
        }
    }
}

fn run_test_nopanic<F>(mut f: F)
    where F: FnMut(&Allocator)
{
    let first_alloc = Allocator::new(usize::MAX);
    f(&first_alloc);
}

fn main() {
    run_test(|a| dynamic_init(a, false));
    run_test(|a| dynamic_init(a, true));
    run_test(|a| dynamic_drop(a, false));
    run_test(|a| dynamic_drop(a, true));

    run_test(|a| assignment2(a, false, false));
    run_test(|a| assignment2(a, false, true));
    run_test(|a| assignment2(a, true, false));
    run_test(|a| assignment2(a, true, true));

    run_test(|a| assignment1(a, false));
    run_test(|a| assignment1(a, true));

    run_test(|a| array_simple(a));
    run_test(|a| vec_simple(a));
    run_test(|a| vec_unreachable(a));

    run_test(|a| struct_dynamic_drop(a, false, false, false));
    run_test(|a| struct_dynamic_drop(a, false, false, true));
    run_test(|a| struct_dynamic_drop(a, false, true, false));
    run_test(|a| struct_dynamic_drop(a, false, true, true));
    run_test(|a| struct_dynamic_drop(a, true, false, false));
    run_test(|a| struct_dynamic_drop(a, true, false, true));
    run_test(|a| struct_dynamic_drop(a, true, true, false));
    run_test(|a| struct_dynamic_drop(a, true, true, true));

    run_test(|a| field_assignment(a, false));
    run_test(|a| field_assignment(a, true));

    run_test(|a| coroutine(a, 0));
    run_test(|a| coroutine(a, 1));
    run_test(|a| coroutine(a, 2));
    run_test(|a| coroutine(a, 3));

    run_test(|a| mixed_drop_and_nondrop(a));

    run_test(|a| slice_pattern_first(a));
    run_test(|a| slice_pattern_middle(a));
    run_test(|a| slice_pattern_two(a));
    run_test(|a| slice_pattern_last(a));
    run_test(|a| slice_pattern_one_of(a, 0));
    run_test(|a| slice_pattern_one_of(a, 1));
    run_test(|a| slice_pattern_one_of(a, 2));
    run_test(|a| slice_pattern_one_of(a, 3));

    run_test(|a| subslice_pattern_from_end(a, true));
    run_test(|a| subslice_pattern_from_end(a, false));
    run_test(|a| subslice_pattern_from_end_with_drop(a, true, true));
    run_test(|a| subslice_pattern_from_end_with_drop(a, true, false));
    run_test(|a| subslice_pattern_from_end_with_drop(a, false, true));
    run_test(|a| subslice_pattern_from_end_with_drop(a, false, false));
    run_test(|a| slice_pattern_reassign(a));
    run_test(|a| subslice_pattern_reassign(a));

    run_test(|a| index_field_mixed_ends(a));
    run_test(|a| subslice_mixed_min_lengths(a, 0));
    run_test(|a| subslice_mixed_min_lengths(a, 1));
    run_test(|a| subslice_mixed_min_lengths(a, 2));
    run_test(|a| subslice_mixed_min_lengths(a, 3));
    run_test(|a| subslice_mixed_min_lengths(a, 4));
    run_test(|a| subslice_mixed_min_lengths(a, 5));
    run_test(|a| subslice_mixed_min_lengths(a, 6));
    run_test(|a| subslice_mixed_min_lengths(a, 7));

    run_test(|a| move_ref_pattern(a));

    run_test(|a| if_let_guard(a, true, 0));
    run_test(|a| if_let_guard(a, true, 1));
    run_test(|a| if_let_guard(a, true, 2));
    run_test(|a| if_let_guard(a, false, 0));
    run_test(|a| if_let_guard(a, false, 1));
    run_test(|a| if_let_guard(a, false, 2));
    run_test(|a| if_let_guard_2(a, 0));
    run_test(|a| if_let_guard_2(a, 1));
    run_test(|a| if_let_guard_2(a, 2));

    run_test(|a| {
        panic_after_return(a);
    });
    run_test(|a| {
        panic_after_return_expr(a);
    });
    run_test(|a| panic_after_init(a));
    run_test(|a| panic_after_init_temp(a));
    run_test(|a| panic_after_init_by_loop(a));

    run_test(|a| bindings_after_at_dynamic_init_move(a, true));
    run_test(|a| bindings_after_at_dynamic_init_move(a, false));
    run_test(|a| bindings_after_at_dynamic_init_ref(a, true));
    run_test(|a| bindings_after_at_dynamic_init_ref(a, false));
    run_test(|a| bindings_after_at_dynamic_drop_move(a, true));
    run_test(|a| bindings_after_at_dynamic_drop_move(a, false));
    run_test(|a| bindings_after_at_dynamic_drop_ref(a, true));
    run_test(|a| bindings_after_at_dynamic_drop_ref(a, false));

    run_test_nopanic(|a| union1(a));
}
