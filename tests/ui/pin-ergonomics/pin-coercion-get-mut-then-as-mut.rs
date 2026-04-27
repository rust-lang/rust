//@ check-pass
//@ edition:2024

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This protects calling `Pin<&mut T>::as_mut()` after using `get_mut()` for
// `T: Unpin`; the `get_mut()` path must not make the later `as_mut()` borrow
// fail as pinned.

fn pin_mut_to_ref<T: Unpin>(value: &pin mut T) -> &T {
    value
}

fn regression<T: Unpin>(value: &mut T) {
    let mut value: &pin mut T = value;
    let _: &mut T = value.get_mut();
    let _: &T = pin_mut_to_ref(value.as_mut());
}

fn main() {
    let mut value = ();
    regression(&mut value);
}
