#![feature(box_syntax)]
#![allow(
    clippy::borrowed_box,
    clippy::needless_pass_by_value,
    clippy::unused_unit,
    clippy::redundant_clone,
    clippy::match_single_binding
)]
#![warn(clippy::boxed_local)]

#[derive(Clone)]
struct A;

impl A {
    fn foo(&self) {}
}

trait Z {
    fn bar(&self);
}

impl Z for A {
    fn bar(&self) {
        //nothing
    }
}

fn main() {}

fn ok_box_trait(boxed_trait: &Box<dyn Z>) {
    let boxed_local = boxed_trait;
    // done
}

fn warn_call() {
    let x = box A;
    x.foo();
}

fn warn_arg(x: Box<A>) {
    x.foo();
}

fn nowarn_closure_arg() {
    let x = Some(box A);
    x.map_or((), |x| take_ref(&x));
}

fn warn_rename_call() {
    let x = box A;

    let y = x;
    y.foo(); // via autoderef
}

fn warn_notuse() {
    let bz = box A;
}

fn warn_pass() {
    let bz = box A;
    take_ref(&bz); // via deref coercion
}

fn nowarn_return() -> Box<A> {
    box A // moved out, "escapes"
}

fn nowarn_move() {
    let bx = box A;
    drop(bx) // moved in, "escapes"
}
fn nowarn_call() {
    let bx = box A;
    bx.clone(); // method only available to Box, not via autoderef
}

fn nowarn_pass() {
    let bx = box A;
    take_box(&bx); // fn needs &Box
}

fn take_box(x: &Box<A>) {}
fn take_ref(x: &A) {}

fn nowarn_ref_take() {
    // false positive, should actually warn
    let x = box A;
    let y = &x;
    take_box(y);
}

fn nowarn_match() {
    let x = box A; // moved into a match
    match x {
        y => drop(y),
    }
}

fn warn_match() {
    let x = box A;
    match &x {
        // not moved
        ref y => (),
    }
}

fn nowarn_large_array() {
    // should not warn, is large array
    // and should not be on stack
    let x = box [1; 10000];
    match &x {
        // not moved
        ref y => (),
    }
}

/// ICE regression test
pub trait Foo {
    type Item;
}

impl<'a> Foo for &'a () {
    type Item = ();
}

pub struct PeekableSeekable<I: Foo> {
    _peeked: I::Item,
}

pub fn new(_needs_name: Box<PeekableSeekable<&()>>) -> () {}

/// Regression for #916, #1123
///
/// This shouldn't warn for `boxed_local`as the implementation of a trait
/// can't change much about the trait definition.
trait BoxedAction {
    fn do_sth(self: Box<Self>);
}

impl BoxedAction for u64 {
    fn do_sth(self: Box<Self>) {
        println!("{}", *self)
    }
}

/// Regression for #1478
///
/// This shouldn't warn for `boxed_local`as self itself is a box type.
trait MyTrait {
    fn do_sth(self);
}

impl<T> MyTrait for Box<T> {
    fn do_sth(self) {}
}

// Issue #3739 - capture in closures
mod issue_3739 {
    use super::A;

    fn consume<T>(_: T) {}
    fn borrow<T>(_: &T) {}

    fn closure_consume(x: Box<A>) {
        let _ = move || {
            consume(x);
        };
    }

    fn closure_borrow(x: Box<A>) {
        let _ = || {
            borrow(&x);
        };
    }
}

/// Issue #5542
///
/// This shouldn't warn for `boxed_local` as it is intended to called from non-Rust code.
pub extern "C" fn do_not_warn_me(_c_pointer: Box<String>) -> () {}

#[rustfmt::skip] // Forces rustfmt to not add ABI
pub extern fn do_not_warn_me_no_abi(_c_pointer: Box<String>) -> () {}

// Issue #4804 - default implementation in trait
mod issue4804 {
    trait DefaultTraitImplTest {
        // don't warn on `self`
        fn default_impl(self: Box<Self>) -> u32 {
            5
        }

        // warn on `x: Box<u32>`
        fn default_impl_x(self: Box<Self>, x: Box<u32>) -> u32 {
            4
        }
    }

    trait WarnTrait {
        // warn on `x: Box<u32>`
        fn foo(x: Box<u32>) {}
    }
}
