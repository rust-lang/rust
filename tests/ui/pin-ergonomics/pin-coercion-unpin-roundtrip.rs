//@ check-pass
//@ edition:2024

#![feature(pin_ergonomics)]

// This protects the `T: Unpin` round trip from `&mut T` to `&pin mut T` and
// back to `&mut T`, which must not leave the reference falsely treated as pinned.

fn to_pin_mut<T: Unpin>(value: &mut T) -> &pin mut T {
    value
}

fn to_mut<T: Unpin>(value: &pin mut T) -> &mut T {
    value
}

fn touch_mut<T>(_: &mut T) {}

fn regression<T: Unpin>(value: &mut T) {
    let value: &pin mut T = to_pin_mut(value);
    let value: &mut T = to_mut(value);
    touch_mut(value);
}

fn main() {
    let mut value = ();
    regression(&mut value);
}
