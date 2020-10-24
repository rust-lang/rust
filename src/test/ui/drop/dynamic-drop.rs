// run-pass
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(generators, generator_trait)]
#![feature(bindings_after_at)]
#![allow(unused_assignments)]
#![allow(unused_variables)]

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ops::Generator;
use std::panic;
use std::pin::Pin;

struct InjectedFailure;

struct Allocator {
    data: RefCell<Vec<bool>>,
    name: &'static str,
    failing_op: usize,
    cur_ops: Cell<usize>,
}

impl panic::UnwindSafe for Allocator {}
impl panic::RefUnwindSafe for Allocator {}

impl Drop for Allocator {
    fn drop(&mut self) {
        let data = self.data.borrow();
        if data.iter().any(|d| *d) {
            panic!("missing free in {:?}: {:?}", self.name, data);
        }
    }
}

impl Allocator {
    fn new(failing_op: usize, name: &'static str) -> Self {
        Allocator {
            failing_op: failing_op,
            cur_ops: Cell::new(0),
            data: RefCell::new(vec![]),
            name,
        }
    }
    fn alloc(&self) -> Ptr<'_> {
        self.cur_ops.set(self.cur_ops.get() + 1);

        if self.cur_ops.get() == self.failing_op {
            panic!(InjectedFailure);
        }

        let mut data = self.data.borrow_mut();
        let addr = data.len();
        data.push(true);
        Ptr(addr, self)
    }
}

struct Ptr<'a>(usize, &'a Allocator);
impl<'a> Drop for Ptr<'a> {
    fn drop(&mut self) {
        match self.1.data.borrow_mut()[self.0] {
            false => panic!("double free in {:?} at index {:?}", self.1.name, self.0),
            ref mut d => *d = false,
        }

        self.1.cur_ops.set(self.1.cur_ops.get() + 1);

        if self.1.cur_ops.get() == self.1.failing_op {
            panic!(InjectedFailure);
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

struct TwoPtrs<'a>(Ptr<'a>, Ptr<'a>);
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

fn generator(a: &Allocator, run_count: usize) {
    assert!(run_count < 4);

    let mut gen = || {
        (a.alloc(), yield a.alloc(), a.alloc(), yield a.alloc());
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
    let [_x, ..] = [a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_middle(a: &Allocator) {
    let [_, _x, _] = [a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_two(a: &Allocator) {
    let [_x, _, _y, _] = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_last(a: &Allocator) {
    let [.., _y] = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
}

fn slice_pattern_one_of(a: &Allocator, i: usize) {
    let array = [a.alloc(), a.alloc(), a.alloc(), a.alloc()];
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
}

fn subslice_pattern_from_end(a: &Allocator, arg: bool) {
    let a = [a.alloc(), a.alloc(), a.alloc()];
    if arg {
        let [.., _x, _] = a;
    } else {
        let [_, _y @ ..] = a;
    }
}

fn subslice_pattern_from_end_with_drop(a: &Allocator, arg: bool, arg2: bool) {
    let a = [a.alloc(), a.alloc(), a.alloc(), a.alloc(), a.alloc()];
    if arg2 {
        drop(a);
        return;
    }

    if arg {
        let [.., _x, _] = a;
    } else {
        let [_, _y @ ..] = a;
    }
}

fn slice_pattern_reassign(a: &Allocator) {
    let mut ar = [a.alloc(), a.alloc()];
    let [_, _x] = ar;
    ar = [a.alloc(), a.alloc()];
    let [.., _y] = ar;
}

fn subslice_pattern_reassign(a: &Allocator) {
    let mut ar = [a.alloc(), a.alloc(), a.alloc()];
    let [_, _, _x] = ar;
    ar = [a.alloc(), a.alloc(), a.alloc()];
    let [_, _y @ ..] = ar;
}

fn index_field_mixed_ends(a: &Allocator) {
    let ar = [(a.alloc(), a.alloc()), (a.alloc(), a.alloc())];
    let [(_x, _), ..] = ar;
    let [(_, _y), _] = ar;
    let [_, (_, _w)] = ar;
    let [.., (_z, _)] = ar;
}

fn subslice_mixed_min_lengths(a: &Allocator, c: i32) {
    let ar = [(a.alloc(), a.alloc()), (a.alloc(), a.alloc())];
    match c {
        0 => {
            let [_x, ..] = ar;
        }
        1 => {
            let [_x, _, ..] = ar;
        }
        2 => {
            let [_x, _] = ar;
        }
        3 => {
            let [(_x, _), _, ..] = ar;
        }
        4 => {
            let [.., (_x, _)] = ar;
        }
        5 => {
            let [.., (_x, _), _] = ar;
        }
        6 => {
            let [_y @ ..] = ar;
        }
        _ => {
            let [_y @ .., _] = ar;
        }
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

fn panic_after_return(a: &Allocator) -> Ptr<'_> {
    a.alloc();
    let p = a.alloc();
    {
        a.alloc();
        let p = a.alloc();
        a.alloc()
    }
}

fn panic_after_return_expr(a: &Allocator) -> Ptr<'_> {
    a.alloc();
    let p = a.alloc();
    {
        a.alloc();
        let q = a.alloc();
        return a.alloc();
    }
}

fn panic_after_init(a: &Allocator) {
    a.alloc();
    let p = a.alloc();
    let q = {
        a.alloc();
        let r = a.alloc();
        a.alloc()
    };
}

fn panic_after_init_temp(a: &Allocator) {
    a.alloc();
    let p = a.alloc();
    {
        a.alloc();
        let r = a.alloc();
        a.alloc()
    };
}

fn panic_after_init_by_loop(a: &Allocator) {
    a.alloc();
    let p = a.alloc();
    let q = loop {
        a.alloc();
        let r = a.alloc();
        break a.alloc();
    };
}

fn panic_after_init_by_match(a: &Allocator, b: bool) {
    a.alloc();
    let p = a.alloc();
    let _ = loop {
        let q = match b {
            true => {
                a.alloc();
                let r = a.alloc();
                a.alloc()
            }
            false => {
                a.alloc();
                let r = a.alloc();
                break a.alloc();
            }
        };
        return;
    };
}

fn panic_after_init_by_match_with_guard(a: &Allocator, b: bool) {
    a.alloc();
    let p = a.alloc();
    let q = match a.alloc() {
        _ if b => {
            a.alloc();
            let r = a.alloc();
            a.alloc()
        }
        _ => {
            a.alloc();
            let r = a.alloc();
            a.alloc()
        }
    };
}

fn panic_after_init_by_match_with_bindings_and_guard(a: &Allocator, b: bool) {
    a.alloc();
    let p = a.alloc();
    let q = match a.alloc() {
        _x if b => {
            a.alloc();
            let r = a.alloc();
            a.alloc()
        }
        _x => {
            a.alloc();
            let r = a.alloc();
            a.alloc()
        }
    };
}

fn panic_after_init_by_match_with_ref_bindings_and_guard(a: &Allocator, b: bool) {
    a.alloc();
    let p = a.alloc();
    let q = match a.alloc() {
        ref _x if b => {
            a.alloc();
            let r = a.alloc();
            a.alloc()
        }
        ref _x => {
            a.alloc();
            let r = a.alloc();
            a.alloc()
        }
    };
}

fn panic_after_init_by_break_if(a: &Allocator, b: bool) {
    a.alloc();
    let p = a.alloc();
    let q = loop {
        let r = a.alloc();
        break if b {
            let s = a.alloc();
            a.alloc()
        } else {
            a.alloc()
        };
    };
}

fn run_test<F>(mut f: F, name: &'static str)
where
    F: FnMut(&Allocator),
{
    let first_alloc = Allocator::new(usize::MAX, name);
    f(&first_alloc);

    for failing_op in 1..first_alloc.cur_ops.get() + 1 {
        let alloc = Allocator::new(failing_op, name);
        let alloc = &alloc;
        let f = panic::AssertUnwindSafe(&mut f);
        let result = panic::catch_unwind(move || {
            f.0(alloc);
        });
        match result {
            Ok(..) => panic!(
                "test executed {} ops but now {}",
                first_alloc.cur_ops.get(),
                alloc.cur_ops.get()
            ),
            Err(e) => {
                if e.downcast_ref::<InjectedFailure>().is_none() {
                    panic::resume_unwind(e);
                }
            }
        }
    }
}

fn run_test_nopanic<F>(mut f: F, name: &'static str)
where
    F: FnMut(&Allocator),
{
    let first_alloc = Allocator::new(usize::MAX, name);
    f(&first_alloc);
}

macro_rules! run_test {
    ($e:expr) => {
        run_test($e, stringify!($e));
    };
}

fn main() {
    run_test!(|a| dynamic_init(a, false));
    run_test!(|a| dynamic_init(a, true));
    run_test!(|a| dynamic_drop(a, false));
    run_test!(|a| dynamic_drop(a, true));

    run_test!(|a| assignment2(a, false, false));
    run_test!(|a| assignment2(a, false, true));
    run_test!(|a| assignment2(a, true, false));
    run_test!(|a| assignment2(a, true, true));

    run_test!(|a| assignment1(a, false));
    run_test!(|a| assignment1(a, true));

    run_test!(|a| array_simple(a));
    run_test!(|a| vec_simple(a));
    run_test!(|a| vec_unreachable(a));

    run_test!(|a| struct_dynamic_drop(a, false, false, false));
    run_test!(|a| struct_dynamic_drop(a, false, false, true));
    run_test!(|a| struct_dynamic_drop(a, false, true, false));
    run_test!(|a| struct_dynamic_drop(a, false, true, true));
    run_test!(|a| struct_dynamic_drop(a, true, false, false));
    run_test!(|a| struct_dynamic_drop(a, true, false, true));
    run_test!(|a| struct_dynamic_drop(a, true, true, false));
    run_test!(|a| struct_dynamic_drop(a, true, true, true));

    run_test!(|a| field_assignment(a, false));
    run_test!(|a| field_assignment(a, true));

    run_test!(|a| generator(a, 0));
    run_test!(|a| generator(a, 1));
    run_test!(|a| generator(a, 2));
    run_test!(|a| generator(a, 3));

    run_test!(|a| mixed_drop_and_nondrop(a));

    run_test!(|a| slice_pattern_first(a));
    run_test!(|a| slice_pattern_middle(a));
    run_test!(|a| slice_pattern_two(a));
    run_test!(|a| slice_pattern_last(a));
    run_test!(|a| slice_pattern_one_of(a, 0));
    run_test!(|a| slice_pattern_one_of(a, 1));
    run_test!(|a| slice_pattern_one_of(a, 2));
    run_test!(|a| slice_pattern_one_of(a, 3));

    run_test!(|a| subslice_pattern_from_end(a, true));
    run_test!(|a| subslice_pattern_from_end(a, false));
    run_test!(|a| subslice_pattern_from_end_with_drop(a, true, true));
    run_test!(|a| subslice_pattern_from_end_with_drop(a, true, false));
    run_test!(|a| subslice_pattern_from_end_with_drop(a, false, true));
    run_test!(|a| subslice_pattern_from_end_with_drop(a, false, false));
    run_test!(|a| slice_pattern_reassign(a));
    run_test!(|a| subslice_pattern_reassign(a));

    run_test!(|a| index_field_mixed_ends(a));
    run_test!(|a| subslice_mixed_min_lengths(a, 0));
    run_test!(|a| subslice_mixed_min_lengths(a, 1));
    run_test!(|a| subslice_mixed_min_lengths(a, 2));
    run_test!(|a| subslice_mixed_min_lengths(a, 3));
    run_test!(|a| subslice_mixed_min_lengths(a, 4));
    run_test!(|a| subslice_mixed_min_lengths(a, 5));
    run_test!(|a| subslice_mixed_min_lengths(a, 6));
    run_test!(|a| subslice_mixed_min_lengths(a, 7));

    run_test!(|a| move_ref_pattern(a));

    run_test!(|a| {
        panic_after_return(a);
    });
    run_test!(|a| {
        panic_after_return_expr(a);
    });
    run_test!(|a| panic_after_init(a));
    run_test!(|a| panic_after_init_temp(a));
    run_test!(|a| panic_after_init_by_loop(a));
    run_test!(|a| panic_after_init_by_match(a, false));
    run_test!(|a| panic_after_init_by_match(a, true));
    run_test!(|a| panic_after_init_by_match_with_guard(a, false));
    run_test!(|a| panic_after_init_by_match_with_guard(a, true));
    run_test!(|a| panic_after_init_by_match_with_bindings_and_guard(a, false));
    run_test!(|a| panic_after_init_by_match_with_bindings_and_guard(a, true));
    run_test!(|a| panic_after_init_by_match_with_ref_bindings_and_guard(a, false));
    run_test!(|a| panic_after_init_by_match_with_ref_bindings_and_guard(a, true));
    run_test!(|a| panic_after_init_by_break_if(a, false));
    run_test!(|a| panic_after_init_by_break_if(a, true));

    run_test!(|a| bindings_after_at_dynamic_init_move(a, true));
    run_test!(|a| bindings_after_at_dynamic_init_move(a, false));
    run_test!(|a| bindings_after_at_dynamic_init_ref(a, true));
    run_test!(|a| bindings_after_at_dynamic_init_ref(a, false));
    run_test!(|a| bindings_after_at_dynamic_drop_move(a, true));
    run_test!(|a| bindings_after_at_dynamic_drop_move(a, false));
    run_test!(|a| bindings_after_at_dynamic_drop_ref(a, true));
    run_test!(|a| bindings_after_at_dynamic_drop_ref(a, false));

    run_test_nopanic(|a| union1(a), "|a| union1(a)");
}
