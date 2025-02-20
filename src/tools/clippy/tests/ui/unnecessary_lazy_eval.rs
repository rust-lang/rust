//@aux-build: proc_macros.rs
#![warn(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::bind_instead_of_map)]
#![allow(clippy::map_identity)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::unnecessary_literal_unwrap)]
#![allow(clippy::unit_arg)]
#![allow(arithmetic_overflow)]
#![allow(unconditional_panic)]

use std::ops::Deref;

extern crate proc_macros;
use proc_macros::with_span;

struct Deep(Option<usize>);

#[derive(Copy, Clone)]
struct SomeStruct {
    some_field: usize,
}

impl SomeStruct {
    fn return_some_field(&self) -> usize {
        self.some_field
    }
}

fn some_call<T: Default>() -> T {
    T::default()
}

struct Issue9427(i32);

impl Drop for Issue9427 {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

struct Issue9427FollowUp;

impl Drop for Issue9427FollowUp {
    fn drop(&mut self) {
        panic!("side effect drop");
    }
}

struct Issue9427Followup2 {
    ptr: *const (),
}
impl Issue9427Followup2 {
    fn from_owned(ptr: *const ()) -> Option<Self> {
        (!ptr.is_null()).then(|| Self { ptr })
    }
}
impl Drop for Issue9427Followup2 {
    fn drop(&mut self) {}
}

struct Issue10437;
impl Deref for Issue10437 {
    type Target = u32;
    fn deref(&self) -> &Self::Target {
        println!("side effect deref");
        &0
    }
}

fn main() {
    let astronomers_pi = 10;
    let ext_arr: [usize; 1] = [2];
    let ext_str = SomeStruct { some_field: 10 };

    let mut opt = Some(42);
    let ext_opt = Some(42);
    let nested_opt = Some(Some(42));
    let nested_tuple_opt = Some(Some((42, 43)));
    let cond = true;

    // Should lint - Option
    let _ = opt.unwrap_or_else(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.unwrap_or_else(|| astronomers_pi);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.unwrap_or_else(|| ext_str.some_field);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.unwrap_or_else(|| ext_arr[0]);
    let _ = opt.and_then(|_| ext_opt);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.or_else(|| ext_opt);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.or_else(|| None);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.get_or_insert_with(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = opt.ok_or_else(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = nested_tuple_opt.unwrap_or_else(|| Some((1, 2)));
    //~^ unnecessary_lazy_evaluations
    let _ = cond.then(|| astronomers_pi);
    //~^ unnecessary_lazy_evaluations
    let _ = true.then(|| -> _ {});
    //~^ unnecessary_lazy_evaluations
    let _ = true.then(|| {});
    //~^ unnecessary_lazy_evaluations

    // Should lint - Builtin deref
    let r = &1;
    let _ = Some(1).unwrap_or_else(|| *r);
    //~^ unnecessary_lazy_evaluations
    let b = Box::new(1);
    let _ = Some(1).unwrap_or_else(|| *b);
    //~^ unnecessary_lazy_evaluations
    // Should lint - Builtin deref through autoderef
    let _ = Some(1).as_ref().unwrap_or_else(|| &r);
    //~^ unnecessary_lazy_evaluations
    let _ = Some(1).as_ref().unwrap_or_else(|| &b);
    //~^ unnecessary_lazy_evaluations

    // Cases when unwrap is not called on a simple variable
    let _ = Some(10).unwrap_or_else(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = Some(10).and_then(|_| ext_opt);
    //~^ unnecessary_lazy_evaluations
    let _: Option<usize> = None.or_else(|| ext_opt);
    //~^ unnecessary_lazy_evaluations
    let _ = None.get_or_insert_with(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _: Result<usize, usize> = None.ok_or_else(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _: Option<usize> = None.or_else(|| None);
    //~^ unnecessary_lazy_evaluations

    let mut deep = Deep(Some(42));
    let _ = deep.0.unwrap_or_else(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = deep.0.and_then(|_| ext_opt);
    //~^ unnecessary_lazy_evaluations
    let _ = deep.0.or_else(|| None);
    //~^ unnecessary_lazy_evaluations
    let _ = deep.0.get_or_insert_with(|| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = deep.0.ok_or_else(|| 2);
    //~^ unnecessary_lazy_evaluations

    // Should not lint - Option
    let _ = opt.unwrap_or_else(|| ext_str.return_some_field());
    let _ = nested_opt.unwrap_or_else(|| Some(some_call()));
    let _ = nested_tuple_opt.unwrap_or_else(|| Some((some_call(), some_call())));
    let _ = opt.or_else(some_call);
    let _ = opt.or_else(|| some_call());
    let _: Result<usize, usize> = opt.ok_or_else(|| some_call());
    let _: Result<usize, usize> = opt.ok_or_else(some_call);
    let _ = deep.0.get_or_insert_with(|| some_call());
    let _ = deep.0.or_else(some_call);
    let _ = deep.0.or_else(|| some_call());
    let _ = opt.ok_or_else(|| ext_arr[0]);

    let _ = Some(1).unwrap_or_else(|| *Issue10437); // Issue10437 has a deref impl
    let _ = Some(1).unwrap_or(*Issue10437);

    let _ = Some(1).as_ref().unwrap_or_else(|| &Issue10437);
    let _ = Some(1).as_ref().unwrap_or(&Issue10437);

    // Should not lint - bool
    let _ = (0 == 1).then(|| Issue9427(0)); // Issue9427 has a significant drop
    let _ = false.then(|| Issue9427FollowUp); // Issue9427FollowUp has a significant drop
    let _ = false.then(|| Issue9427Followup2 { ptr: std::ptr::null() });

    // should not lint, bind_instead_of_map takes priority
    let _ = Some(10).and_then(|idx| Some(ext_arr[idx]));
    let _ = Some(10).and_then(|idx| Some(idx));

    // should lint, bind_instead_of_map doesn't apply
    let _: Option<usize> = None.or_else(|| Some(3));
    //~^ unnecessary_lazy_evaluations
    let _ = deep.0.or_else(|| Some(3));
    //~^ unnecessary_lazy_evaluations
    let _ = opt.or_else(|| Some(3));
    //~^ unnecessary_lazy_evaluations

    // Should lint - Result
    let res: Result<usize, usize> = Err(5);
    let res2: Result<usize, SomeStruct> = Err(SomeStruct { some_field: 5 });

    let _ = res2.unwrap_or_else(|_| 2);
    //~^ unnecessary_lazy_evaluations
    let _ = res2.unwrap_or_else(|_| astronomers_pi);
    //~^ unnecessary_lazy_evaluations
    let _ = res2.unwrap_or_else(|_| ext_str.some_field);
    //~^ unnecessary_lazy_evaluations

    // Should not lint - Result
    let _ = res.unwrap_or_else(|err| err);
    let _ = res.unwrap_or_else(|err| ext_arr[err]);
    let _ = res2.unwrap_or_else(|err| err.some_field);
    let _ = res2.unwrap_or_else(|err| err.return_some_field());
    let _ = res2.unwrap_or_else(|_| ext_str.return_some_field());

    // should not lint, bind_instead_of_map takes priority
    let _: Result<usize, usize> = res.and_then(|x| Ok(x));
    let _: Result<usize, usize> = res.or_else(|err| Err(err));

    let _: Result<usize, usize> = res.and_then(|_| Ok(2));
    let _: Result<usize, usize> = res.and_then(|_| Ok(astronomers_pi));
    let _: Result<usize, usize> = res.and_then(|_| Ok(ext_str.some_field));

    let _: Result<usize, usize> = res.or_else(|_| Err(2));
    let _: Result<usize, usize> = res.or_else(|_| Err(astronomers_pi));
    let _: Result<usize, usize> = res.or_else(|_| Err(ext_str.some_field));

    // should lint, bind_instead_of_map doesn't apply
    let _: Result<usize, usize> = res.and_then(|_| Err(2));
    //~^ unnecessary_lazy_evaluations
    let _: Result<usize, usize> = res.and_then(|_| Err(astronomers_pi));
    //~^ unnecessary_lazy_evaluations
    let _: Result<usize, usize> = res.and_then(|_| Err(ext_str.some_field));
    //~^ unnecessary_lazy_evaluations

    let _: Result<usize, usize> = res.or_else(|_| Ok(2));
    //~^ unnecessary_lazy_evaluations
    let _: Result<usize, usize> = res.or_else(|_| Ok(astronomers_pi));
    //~^ unnecessary_lazy_evaluations
    let _: Result<usize, usize> = res.or_else(|_| Ok(ext_str.some_field));
    //~^ unnecessary_lazy_evaluations
    let _: Result<usize, usize> = res.
    //~^ unnecessary_lazy_evaluations
    // some lines
    // some lines
    // some lines
    // some lines
    // some lines
    // some lines
    or_else(|_| Ok(ext_str.some_field));

    // neither bind_instead_of_map nor unnecessary_lazy_eval applies here
    let _: Result<usize, usize> = res.and_then(|x| Err(x));
    let _: Result<usize, usize> = res.or_else(|err| Ok(err));
}

#[allow(unused)]
fn issue9485() {
    // should not lint, is in proc macro
    with_span!(span Some(42).unwrap_or_else(|| 2););
}

fn issue9422(x: usize) -> Option<usize> {
    (x >= 5).then(|| x - 5)
    // (x >= 5).then_some(x - 5)  // clippy suggestion panics
}

fn panicky_arithmetic_ops(x: usize, y: isize) {
    #![allow(clippy::identity_op, clippy::eq_op)]

    // See comments in `eager_or_lazy.rs` for the rules that this is meant to follow

    let _x = false.then(|| i32::MAX + 1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| i32::MAX * 2);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| i32::MAX - 1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| i32::MIN - 1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| (1 + 2 * 3 - 2 / 3 + 9) << 2);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 255u8 << 7);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 255u8 << 8);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 255u8 >> 8);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 255u8 >> x);
    let _x = false.then(|| i32::MAX + -1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| -i32::MAX);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| -i32::MIN);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| -y);
    let _x = false.then(|| 255 >> -7);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 255 << -1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 1 / 0);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| x << -1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| x << 2);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| x + x);
    let _x = false.then(|| x * x);
    let _x = false.then(|| x - x);
    let _x = false.then(|| x / x);
    let _x = false.then(|| x % x);
    let _x = false.then(|| x + 1);
    let _x = false.then(|| 1 + x);

    let _x = false.then(|| x / 0);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| x % 0);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| y / -1);
    let _x = false.then(|| 1 / -1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| i32::MIN / -1);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| i32::MIN / x as i32);
    let _x = false.then(|| i32::MIN / 0);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 4 / 2);
    //~^ unnecessary_lazy_evaluations
    let _x = false.then(|| 1 / x);

    // const eval doesn't read variables, but floating point math never panics, so we can still emit a
    // warning
    let f1 = 1.0;
    let f2 = 2.0;
    let _x = false.then(|| f1 + f2);
    //~^ unnecessary_lazy_evaluations
}
