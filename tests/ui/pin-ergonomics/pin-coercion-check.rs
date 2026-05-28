//@ revisions: pin_ergonomics normal
//@ edition:2024
#![cfg_attr(pin_ergonomics, feature(pin_ergonomics))]
#![allow(incomplete_features)]

use std::pin::Pin;

fn get<T, U: Unpin>() {
    |x: Pin<&mut T>| -> &mut T { x.get_mut() }; //~ ERROR `T` cannot be unpinned
    |x: Pin<&T>| -> &T { x.get_ref() };

    |x: Pin<&mut U>| -> &mut U { x.get_mut() };
    |x: Pin<&U>| -> &U { x.get_ref() };
}

fn pin_to_ref<T, U: Unpin>() {
    // T: !Unpin
    |x: Pin<&mut T>| -> &mut T { x };
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR `T` cannot be unpinned
    |x: Pin<&mut T>| -> &T { x };
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR `T` cannot be unpinned
    |x: Pin<&T>| -> &T { x };
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR `T` cannot be unpinned
    |x: Pin<&T>| -> &mut T { x };
    //~^ ERROR mismatched types

    // U: Unpin
    |x: Pin<&mut U>| -> &mut U { x };
    //[normal]~^ ERROR mismatched types
    |x: Pin<&mut U>| -> &U { x };
    //[normal]~^ ERROR mismatched types
    |x: Pin<&U>| -> &U { x };
    //[normal]~^ ERROR mismatched types
    |x: Pin<&U>| -> &mut U { x };
    //~^ ERROR mismatched types
}

fn ref_to_pin<T, U: Unpin>() {
    // T: !Unpin
    |x: &mut T| -> Pin<&mut T> { x };
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR `T` cannot be unpinned
    |x: &mut T| -> Pin<&T> { x };
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR `T` cannot be unpinned
    |x: &T| -> Pin<&T> { x };
    //[normal]~^ ERROR mismatched types
    //[pin_ergonomics]~^^ ERROR `T` cannot be unpinned
    |x: &T| -> Pin<&mut T> { x };
    //~^ ERROR mismatched types

    // U: Unpin
    |x: &mut U| -> Pin<&mut U> { x };
    //[normal]~^ ERROR mismatched types
    |x: &mut U| -> Pin<&U> { x };
    //[normal]~^ ERROR mismatched types
    |x: &U| -> Pin<&U> { x };
    //[normal]~^ ERROR mismatched types
    |x: &U| -> Pin<&mut U> { x };
    //~^ ERROR mismatched types
}

fn main() {}
