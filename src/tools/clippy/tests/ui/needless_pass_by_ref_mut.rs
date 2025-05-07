#![allow(
    clippy::if_same_then_else,
    clippy::no_effect,
    clippy::ptr_arg,
    clippy::redundant_closure_call,
    clippy::uninlined_format_args
)]
#![warn(clippy::needless_pass_by_ref_mut)]
//@no-rustfix
use std::ptr::NonNull;

fn foo(s: &mut Vec<u32>, b: &u32, x: &mut u32) {
    //~^ needless_pass_by_ref_mut

    *x += *b + s.len() as u32;
}

// Should not warn.
fn foo2(s: &mut Vec<u32>) {
    s.push(8);
}

// Should not warn because we return it.
fn foo3(s: &mut Vec<u32>) -> &mut Vec<u32> {
    s
}

// Should not warn because `s` is used as mutable.
fn foo4(s: &mut Vec<u32>) {
    Vec::push(s, 4);
}

// Should not warn.
fn foo5(s: &mut Vec<u32>) {
    foo2(s);
}

fn foo6(s: &mut Vec<u32>) {
    //~^ needless_pass_by_ref_mut

    non_mut_ref(s);
}

fn non_mut_ref(_: &Vec<u32>) {}

struct Bar;

impl Bar {
    fn bar(&mut self) {}
    //~^ needless_pass_by_ref_mut

    fn mushroom(&self, vec: &mut Vec<i32>) -> usize {
        //~^ needless_pass_by_ref_mut

        vec.len()
    }
}

trait Babar {
    // Should not warn here since it's a trait method.
    fn method(arg: &mut u32);
}

impl Babar for Bar {
    // Should not warn here since it's a trait method.
    fn method(a: &mut u32) {}
}

// Should not warn (checking variable aliasing).
fn alias_check(s: &mut Vec<u32>) {
    let mut alias = s;
    let mut alias2 = alias;
    let mut alias3 = alias2;
    alias3.push(0);
}

// Should not warn (checking variable aliasing).
fn alias_check2(mut s: &mut Vec<u32>) {
    let mut alias = &mut s;
    alias.push(0);
}

struct Mut<T> {
    ptr: NonNull<T>,
}

impl<T> Mut<T> {
    // Should not warn because `NonNull::from` also accepts `&mut`.
    fn new(ptr: &mut T) -> Self {
        Mut {
            ptr: NonNull::from(ptr),
        }
    }
}

// Should not warn.
fn unused(_: &mut u32, _b: &mut u8) {}

// Should not warn.
async fn f1(x: &mut i32) {
    *x += 1;
}
// Should not warn.
async fn f2(x: &mut i32, y: String) {
    *x += 1;
}
// Should not warn.
async fn f3(x: &mut i32, y: String, z: String) {
    *x += 1;
}
// Should not warn.
async fn f4(x: &mut i32, y: i32) {
    *x += 1;
}
// Should not warn.
async fn f5(x: i32, y: &mut i32) {
    *y += 1;
}
// Should not warn.
async fn f6(x: i32, y: &mut i32, z: &mut i32) {
    *y += 1;
    *z += 1;
}
// Should not warn.
async fn f7(x: &mut i32, y: i32, z: &mut i32, a: i32) {
    *x += 1;
    *z += 1;
}

async fn a1(x: &mut i32) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", x);
}
async fn a2(x: &mut i32, y: String) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", x);
}
async fn a3(x: &mut i32, y: String, z: String) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", x);
}
async fn a4(x: &mut i32, y: i32) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", x);
}
async fn a5(x: i32, y: &mut i32) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", x);
}
async fn a6(x: i32, y: &mut i32) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", x);
}
async fn a7(x: i32, y: i32, z: &mut i32) {
    //~^ needless_pass_by_ref_mut

    println!("{:?}", z);
}
async fn a8(x: i32, a: &mut i32, y: i32, z: &mut i32) {
    //~^ needless_pass_by_ref_mut
    //~| needless_pass_by_ref_mut

    println!("{:?}", z);
}

// Should not warn (passed as closure which takes `&mut`).
fn passed_as_closure(s: &mut u32) {}

// Should not warn.
fn passed_as_local(s: &mut u32) {}

// Should not warn.
fn ty_unify_1(s: &mut u32) {}

// Should not warn.
fn ty_unify_2(s: &mut u32) {}

// Should not warn.
fn passed_as_field(s: &mut u32) {}

fn closure_takes_mut(s: fn(&mut u32)) {}

struct A {
    s: fn(&mut u32),
}

// Should warn.
fn used_as_path(s: &mut u32) {}

// Make sure lint attributes work fine
#[expect(clippy::needless_pass_by_ref_mut)]
fn lint_attr(s: &mut u32) {}

#[cfg(not(feature = "a"))]
fn cfg_warn(s: &mut u32) {}
//~^ needless_pass_by_ref_mut

#[cfg(not(feature = "a"))]
mod foo {
    fn cfg_warn(s: &mut u32) {}
    //~^ needless_pass_by_ref_mut
}

// Should not warn.
async fn inner_async(x: &mut i32, y: &mut u32) {
    async {
        *y += 1;
        *x += 1;
    }
    .await;
}

async fn inner_async2(x: &mut i32, y: &mut u32) {
    //~^ needless_pass_by_ref_mut

    async {
        *x += 1;
    }
    .await;
}

async fn inner_async3(x: &mut i32, y: &mut u32) {
    //~^ needless_pass_by_ref_mut

    async {
        *y += 1;
    }
    .await;
}

// Should not warn.
async fn async_vec(b: &mut Vec<bool>) {
    b.append(&mut vec![]);
}

// Should not warn.
async fn async_vec2(b: &mut Vec<bool>) {
    b.push(true);
}
fn non_mut(n: &str) {}
//Should warn
async fn call_in_closure1(n: &mut str) {
    //~^ needless_pass_by_ref_mut
    (|| non_mut(n))()
}
fn str_mut(str: &mut String) -> bool {
    str.pop().is_some()
}
//Should not warn
async fn call_in_closure2(str: &mut String) {
    (|| str_mut(str))();
}

// Should not warn.
async fn closure(n: &mut usize) -> impl '_ + FnMut() {
    || {
        *n += 1;
    }
}

// Should warn.
fn closure2(n: &mut usize) -> impl '_ + FnMut() -> usize {
    //~^ needless_pass_by_ref_mut

    || *n + 1
}

// Should not warn.
async fn closure3(n: &mut usize) {
    (|| *n += 1)();
}

// Should warn.
async fn closure4(n: &mut usize) {
    //~^ needless_pass_by_ref_mut

    (|| {
        let _x = *n + 1;
    })();
}

// Should not warn: pub
pub fn pub_foo(s: &mut Vec<u32>, b: &u32, x: &mut u32) {
    *x += *b + s.len() as u32;
}

// Should not warn.
async fn _f(v: &mut Vec<()>) {
    let x = || v.pop();
    _ = || || x;
}

struct Data<T: ?Sized> {
    value: T,
}
// Unsafe functions should not warn.
unsafe fn get_mut_unchecked<T>(ptr: &mut NonNull<Data<T>>) -> &mut T {
    unsafe { &mut (*ptr.as_ptr()).value }
}
// Unsafe blocks should not warn.
fn get_mut_unchecked2<T>(ptr: &mut NonNull<Data<T>>) -> &mut T {
    unsafe { &mut (*ptr.as_ptr()).value }
}

fn set_true(b: &mut bool) {
    *b = true;
}

// Should not warn.
fn true_setter(b: &mut bool) -> impl FnOnce() + '_ {
    move || set_true(b)
}

// Should not warn.
fn filter_copy<T: Copy>(predicate: &mut impl FnMut(T) -> bool) -> impl FnMut(&T) -> bool + '_ {
    move |&item| predicate(item)
}

trait MutSelfTrait {
    // Should not warn since it's a trait method.
    fn mut_self(&mut self);
}

struct MutSelf {
    a: u32,
}

impl MutSelf {
    fn bar(&mut self) {}
    //~^ needless_pass_by_ref_mut

    async fn foo(&mut self, u: &mut i32, v: &mut u32) {
        //~^ needless_pass_by_ref_mut
        //~| needless_pass_by_ref_mut

        async {
            *u += 1;
        }
        .await;
    }
    async fn foo2(&mut self, u: &mut i32, v: &mut u32) {
        //~^ needless_pass_by_ref_mut

        async {
            self.a += 1;
            *u += 1;
        }
        .await;
    }
}

impl MutSelfTrait for MutSelf {
    // Should not warn since it's a trait method.
    fn mut_self(&mut self) {}
}

// `is_from_proc_macro` stress tests
fn _empty_tup(x: &mut (())) {}
//~^ needless_pass_by_ref_mut
fn _single_tup(x: &mut ((i32,))) {}
//~^ needless_pass_by_ref_mut
fn _multi_tup(x: &mut ((i32, u32))) {}
//~^ needless_pass_by_ref_mut
fn _fn(x: &mut (fn())) {}
//~^ needless_pass_by_ref_mut
#[rustfmt::skip]
fn _extern_rust_fn(x: &mut extern "Rust" fn()) {}
//~^ needless_pass_by_ref_mut
fn _extern_c_fn(x: &mut extern "C" fn()) {}
//~^ needless_pass_by_ref_mut
fn _unsafe_fn(x: &mut unsafe fn()) {}
//~^ needless_pass_by_ref_mut
fn _unsafe_extern_fn(x: &mut unsafe extern "C" fn()) {}
//~^ needless_pass_by_ref_mut
fn _fn_with_arg(x: &mut unsafe extern "C" fn(i32)) {}
//~^ needless_pass_by_ref_mut
fn _fn_with_ret(x: &mut unsafe extern "C" fn() -> (i32)) {}
//~^ needless_pass_by_ref_mut

fn main() {
    let mut u = 0;
    let mut v = vec![0];
    foo(&mut v, &0, &mut u);
    foo2(&mut v);
    foo3(&mut v);
    foo4(&mut v);
    foo5(&mut v);
    alias_check(&mut v);
    alias_check2(&mut v);
    println!("{u}");
    closure_takes_mut(passed_as_closure);
    A { s: passed_as_field };
    used_as_path;
    let _: fn(&mut u32) = passed_as_local;
    let _ = if v[0] == 0 { ty_unify_1 } else { ty_unify_2 };
    pub_foo(&mut v, &0, &mut u);
}
