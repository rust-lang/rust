//@ check-pass
//@ run-rustfix
//
// rust-lang/rust#73592: borrow_mut through Deref should work.
//
// Before #72280, when we see something like `&mut *rcvr.method()`, we
// incorrectly requires `rcvr` to be type-checked as a mut place. While this
// requirement is usually correct for smart pointers, it is overly restrictive
// for types like `Mutex` or `RefCell` which can produce a guard that
// implements `DerefMut` from `&self`.
//
// Making it more confusing, because we use Deref as the fallback when DerefMut
// is implemented, we won't see an issue when the smart pointer does not
// implement `DerefMut`. It only causes an issue when `rcvr` is obtained via a
// type that implements both `Deref` or `DerefMut`.
//
// This bug is only discovered in #73592 after it is already fixed as a side-effect
// of a refactoring made in #72280.

#![warn(unused_mut)]

use std::pin::Pin;
use std::cell::RefCell;

struct S(RefCell<()>);

fn test_pin(s: Pin<&S>) {
    // This works before #72280.
    let _ = &mut *s.0.borrow_mut();
}

fn test_pin_mut(s: Pin<&mut S>) {
    // This should compile but didn't before #72280.
    let _ = &mut *s.0.borrow_mut();
}

fn test_vec(s: &Vec<RefCell<()>>) {
    // This should compile but didn't before #72280.
    let _ = &mut *s[0].borrow_mut();
}

fn test_mut_pin(mut s: Pin<&S>) {
    //~^ WARN variable does not need to be mutable
    let _ = &mut *s.0.borrow_mut();
}

fn test_mut_pin_mut(mut s: Pin<&mut S>) {
    //~^ WARN variable does not need to be mutable
    let _ = &mut *s.0.borrow_mut();
}

fn main() {
    let mut s = S(RefCell::new(()));
    test_pin(Pin::new(&s));
    test_pin_mut(Pin::new(&mut s));
    test_mut_pin(Pin::new(&s));
    test_mut_pin_mut(Pin::new(&mut s));
    test_vec(&vec![s.0]);
}
