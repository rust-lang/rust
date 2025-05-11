//@ check-pass
//! False-positive tests to ensure we don't suggest `const` for things where it would cause a
//! compilation error.
//! The .stderr output of this test should be empty. Otherwise it's a bug somewhere.

//@aux-build:helper.rs
//@aux-build:../auxiliary/proc_macros.rs

#![warn(clippy::missing_const_for_fn)]
#![feature(type_alias_impl_trait)]

extern crate helper;
extern crate proc_macros;

use proc_macros::with_span;

struct Game; // You just lost.

// This should not be linted because it's already const
const fn already_const() -> i32 {
    32
}

impl Game {
    // This should not be linted because it's already const
    pub const fn already_const() -> i32 {
        32
    }
}

// Allowing on this function, because it would lint, which we don't want in this case.
#[allow(clippy::missing_const_for_fn)]
fn random() -> u32 {
    42
}

// We should not suggest to make this function `const` because `random()` is non-const
fn random_caller() -> u32 {
    random()
}

static Y: u32 = 0;

// We should not suggest to make this function `const` because const functions are not allowed to
// refer to a static variable
fn get_y() -> u32 {
    Y
}

#[cfg(test)]
mod with_test_fn {
    #[derive(Clone, Copy)]
    pub struct Foo {
        pub n: u32,
    }

    impl Foo {
        #[must_use]
        pub const fn new(n: u32) -> Foo {
            Foo { n }
        }
    }

    #[test]
    fn foo_is_copy() {
        let foo = Foo::new(42);
        let one = foo;
        let two = foo;
        _ = one;
        _ = two;
    }
}

trait Foo {
    // This should not be suggested to be made const
    // (rustc doesn't allow const trait methods)
    fn f() -> u32;

    // This should not be suggested to be made const either
    fn g() -> u32 {
        33
    }
}

// Don't lint in external macros (derive)
#[derive(PartialEq, Eq)]
struct Point(isize, isize);

impl std::ops::Add for Point {
    type Output = Self;

    // Don't lint in trait impls of derived methods
    fn add(self, other: Self) -> Self {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

mod with_drop {
    pub struct A;
    pub struct B;
    impl Drop for A {
        fn drop(&mut self) {}
    }

    impl A {
        // This can not be const because the type implements `Drop`.
        pub fn b(self) -> B {
            B
        }
    }

    impl B {
        // This can not be const because `a` implements `Drop`.
        pub fn a(self, a: A) -> B {
            B
        }
    }
}

fn const_generic_params<T, const N: usize>(t: &[T; N]) -> &[T; N] {
    t
}

fn const_generic_return<T, const N: usize>(t: &[T]) -> &[T; N] {
    let p = t.as_ptr() as *const [T; N];

    unsafe { &*p }
}

// Do not lint this because it calls a function whose constness is unstable.
fn unstably_const_fn() {
    helper::unstably_const_fn()
}

#[clippy::msrv = "1.46.0"]
mod const_fn_stabilized_after_msrv {
    // Do not lint this because `u8::is_ascii_digit` is stabilized as a const function in 1.47.0.
    fn const_fn_stabilized_after_msrv(byte: u8) {
        byte.is_ascii_digit();
    }
}

with_span! {
    span
    fn dont_check_in_proc_macro() {}
}

// Do not lint `String` has `Vec<u8>`, which cannot be dropped in const contexts
fn a(this: String) {}

enum A {
    F(String),
    N,
}

// Same here.
fn b(this: A) {}

// Minimized version of `a`.
fn c(this: Vec<u16>) {}

struct F(A);

// Do not lint
fn f(this: F) {}

// Do not lint
fn g<T>(this: T) {}

struct Issue10617(String);

impl Issue10617 {
    // Do not lint
    pub fn name(self) -> String {
        self.0
    }
}

union U {
    f: u32,
}

// Do not lint because accessing union fields from const functions is unstable in 1.55
#[clippy::msrv = "1.55"]
fn h(u: U) -> u32 {
    unsafe { u.f }
}

mod msrv {
    struct Foo(*const u8, *mut u8);

    impl Foo {
        #[clippy::msrv = "1.57"]
        fn deref_ptr_cannot_be_const(self) -> usize {
            unsafe { *self.0 as usize }
        }
        #[clippy::msrv = "1.58"]
        fn deref_mut_ptr_cannot_be_const(self) -> usize {
            unsafe { *self.1 as usize }
        }
    }

    #[clippy::msrv = "1.61"]
    extern "C" fn c() {}
}

mod with_ty_alias {
    type Foo = impl std::fmt::Debug;

    #[define_opaque(Foo)]
    fn foo(_: Foo) {
        let _: Foo = 1;
    }
}

// Do not lint because mutable references in const functions are unstable in 1.82
#[clippy::msrv = "1.82"]
fn mut_add(x: &mut i32) {
    *x += 1;
}

#[clippy::msrv = "1.87"]
mod issue14020 {
    use std::ops::Add;

    fn f<T: Add>(a: T, b: T) -> <T as Add>::Output {
        a + b
    }
}

#[clippy::msrv = "1.87"]
mod issue14290 {
    use std::ops::{Deref, DerefMut};

    struct Wrapper<T> {
        t: T,
    }

    impl<T> Deref for Wrapper<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.t
        }
    }
    impl<T> DerefMut for Wrapper<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.t
        }
    }

    struct Example(bool);

    fn do_something(mut a: Wrapper<Example>) {
        a.0 = !a.0;
    }

    pub struct Stream(Vec<u8>);

    impl Stream {
        pub fn bytes(&self) -> &[u8] {
            &self.0
        }
    }
}

#[clippy::msrv = "1.87"]
mod issue14091 {
    use std::mem::ManuallyDrop;

    struct BucketSlotGuard<'a> {
        id: u32,
        free_list: &'a mut Vec<u32>,
    }

    impl BucketSlotGuard<'_> {
        fn into_inner(self) -> u32 {
            let this = ManuallyDrop::new(self);
            this.id
        }
    }

    use std::ops::{Deref, DerefMut};

    struct Wrap<T>(T);

    impl<T> Deref for Wrap<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.0
        }
    }

    impl<T> DerefMut for Wrap<T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }

    fn smart_two_field(v: &mut Wrap<(i32, i32)>) {
        let _a = &mut v.0;
        let _b = &mut v.1;
    }

    fn smart_destructure(v: &mut Wrap<(i32, i32)>) {
        let (ref mut _head, ref mut _tail) = **v;
    }
}
