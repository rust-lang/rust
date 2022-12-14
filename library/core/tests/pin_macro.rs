// edition:2021
use core::{
    marker::PhantomPinned,
    mem::{drop as stuff, transmute},
    pin::{pin, Pin},
};

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
