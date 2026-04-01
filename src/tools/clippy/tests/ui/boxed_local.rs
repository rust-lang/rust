#![allow(
    clippy::borrowed_box,
    clippy::needless_pass_by_value,
    clippy::unused_unit,
    clippy::redundant_clone,
    clippy::match_single_binding,
    clippy::needless_lifetimes
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
    let x = Box::new(A);
    x.foo();
}

fn warn_arg(x: Box<A>) {
    //~^ boxed_local

    x.foo();
}

fn nowarn_closure_arg() {
    let x = Some(Box::new(A));
    x.map_or((), |x| take_ref(&x));
}

fn warn_rename_call() {
    let x = Box::new(A);

    let y = x;
    y.foo(); // via autoderef
}

fn warn_notuse() {
    let bz = Box::new(A);
}

fn warn_pass() {
    let bz = Box::new(A);
    take_ref(&bz); // via deref coercion
}

fn nowarn_return() -> Box<A> {
    Box::new(A) // moved out, "escapes"
}

fn nowarn_move() {
    let bx = Box::new(A);
    drop(bx) // moved in, "escapes"
}
fn nowarn_call() {
    let bx = Box::new(A);
    bx.clone(); // method only available to Box, not via autoderef
}

fn nowarn_pass() {
    let bx = Box::new(A);
    take_box(&bx); // fn needs &Box
}

fn take_box(x: &Box<A>) {}
fn take_ref(x: &A) {}

fn nowarn_ref_take() {
    // false positive, should actually warn
    let x = Box::new(A);
    let y = &x;
    take_box(y);
}

fn nowarn_match() {
    let x = Box::new(A); // moved into a match
    match x {
        y => drop(y),
    }
}

fn warn_match() {
    let x = Box::new(A);
    match &x {
        // not moved
        y => (),
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
//~^ boxed_local

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

#[allow(missing_abi)]
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
            //~^ boxed_local

            4
        }
    }

    trait WarnTrait {
        // warn on `x: Box<u32>`
        fn foo(x: Box<u32>) {}
        //~^ boxed_local
    }
}

fn check_expect(#[expect(clippy::boxed_local)] x: Box<A>) {
    x.foo();
}
