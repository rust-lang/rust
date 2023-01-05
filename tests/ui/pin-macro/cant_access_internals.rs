// edition:2018
#![feature(pin_macro)]

use core::{
    marker::PhantomPinned,
    mem,
    pin::{pin, Pin},
};

fn main() {
    let mut phantom_pinned = pin!(PhantomPinned);
    mem::take(phantom_pinned.pointer); //~ ERROR use of unstable library feature 'unsafe_pin_internals'
}
