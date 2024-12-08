//@ edition:2018

use core::{
    marker::PhantomPinned,
    mem,
    pin::{pin, Pin},
};

fn main() {
    let mut phantom_pinned = pin!(PhantomPinned);
    mem::take(phantom_pinned.__pointer); //~ ERROR use of unstable library feature `unsafe_pin_internals`
}
