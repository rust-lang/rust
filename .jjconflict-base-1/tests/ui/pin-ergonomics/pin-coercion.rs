//@ run-pass
//@ edition:2024
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
#![deny(dead_code)]

use std::cell::RefCell;

fn coerce_mut_to_pin_mut<T: Unpin>(x: &mut T) -> &pin mut T {
    x
}
fn coerce_ref_to_pin_ref<T: Unpin>(x: &T) -> &pin const T {
    x
}
fn coerce_pin_mut_to_mut<T: Unpin>(x: &pin mut T) -> &mut T {
    x
}
fn coerce_pin_ref_to_ref<T: Unpin>(x: &pin const T) -> &T {
    x
}

fn coerce_pin_mut_to_ref<T: Unpin>(x: &pin mut T) -> &T {
    x
}
fn coerce_mut_to_pin_ref<T: Unpin>(x: &mut T) -> &pin const T {
    x
}

fn test(x: &mut RefCell<String>) {
    let mut x: &pin mut _ = coerce_mut_to_pin_mut(x);
    x.get_mut().get_mut().push_str("&mut T -> &pin mut T\n");
    let x_ref: &_ = coerce_pin_mut_to_ref(x.as_mut());
    x_ref.borrow_mut().push_str("&pin mut T -> &T\n");
    let x: &mut _ = coerce_pin_mut_to_mut(x);
    x.get_mut().push_str("&pin mut T -> &mut T\n");
    let x: &pin const _ = coerce_mut_to_pin_ref(x);
    x.borrow_mut().push_str("&mut T -> &pin const T\n");
    let x: &_ = coerce_pin_ref_to_ref(x);
    x.borrow_mut().push_str("&pin const T -> &T\n");
    let x: &pin const _ = coerce_ref_to_pin_ref(x);
    x.borrow_mut().push_str("&T -> &pin const T\n");
}

fn main() {
    let mut x = RefCell::new(String::new());
    test(&mut x);
    assert_eq!(
        x.borrow().as_str(),
        "&mut T -> &pin mut T\n\
        &pin mut T -> &T\n\
        &pin mut T -> &mut T\n\
        &mut T -> &pin const T\n\
        &pin const T -> &T\n\
        &T -> &pin const T\n"
    );
}
