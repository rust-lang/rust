// edition:2018
#![feature(pin_macro)]

use core::{
    convert::identity,
    marker::PhantomPinned,
    mem::drop as stuff,
    pin::pin,
};

fn function_call_stops_borrow_extension() {
    let phantom_pinned = identity(pin!(PhantomPinned));
    //~^ ERROR temporary value dropped while borrowed
    stuff(phantom_pinned)
}

fn promotion_only_works_for_the_innermost_block() {
    let phantom_pinned = {
        let phantom_pinned = pin!(PhantomPinned);
        //~^ ERROR temporary value dropped while borrowed
        phantom_pinned
    };
    stuff(phantom_pinned)
}

fn main() {
    function_call_stops_borrow_extension();
    promotion_only_works_for_the_innermost_block();
}
