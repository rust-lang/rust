// edition:2021

use core::marker::PhantomPinned;
use core::mem::{drop as stuff, transmute};
use core::pin::{Pin, pin};

#[test]
fn basic() {
    let it: Pin<&mut PhantomPinned> = pin!(PhantomPinned);
    stuff(it);
}

#[test]
fn extension_works_through_block() {
    let it: Pin<&mut PhantomPinned> = { pin!(PhantomPinned) };
    stuff(it);
}

#[test]
fn extension_works_through_unsafe_block() {
    // "retro-type-inference" works as well.
    let it: Pin<&mut PhantomPinned> = unsafe { pin!(transmute(())) };
    stuff(it);
}

#[test]
fn unsize_coercion() {
    let slice: Pin<&mut [PhantomPinned]> = pin!([PhantomPinned; 2]);
    stuff(slice);
    let dyn_obj: Pin<&mut dyn Send> = pin!([PhantomPinned; 2]);
    stuff(dyn_obj);
}

#[test]
fn rust_2024_expr() {
    // Check that we accept a Rust 2024 $expr.
    std::pin::pin!(const { 1 });
}

#[test]
fn temp_lifetime() {
    // Check that temporary lifetimes work as in Rust 2021.
    // Regression test for https://github.com/rust-lang/rust/issues/138596
    match std::pin::pin!(foo(&mut 0)) {
        _ => {}
    }
    async fn foo(_: &mut usize) {}
}

#[test]
fn transitive_extension() {
    async fn temporary() {}

    // `pin!` witnessed in the wild being used like this, even if it yields
    // a `Pin<&mut &mut impl Unpin>`; it does work because `pin!`
    // happens to transitively extend the lifespan of `temporary()`.
    let p = pin!(&mut temporary());
    let _use = p;
}
