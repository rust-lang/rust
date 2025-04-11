//@ edition:2018

use core::{
    convert::identity,
    marker::PhantomPinned,
    mem::drop as stuff,
    pin::pin,
};

fn function_call_stops_borrow_extension() {
    let phantom_pinned = identity(pin!(PhantomPinned));
    //~^ ERROR does not live long enough
    stuff(phantom_pinned)
}

fn promotion_only_works_for_the_innermost_block() {
    let phantom_pinned = {
        let phantom_pinned = pin!(PhantomPinned);
        //~^ ERROR does not live long enough
        phantom_pinned
    };
    stuff(phantom_pinned)
}

fn main() {
    function_call_stops_borrow_extension();
    promotion_only_works_for_the_innermost_block();
}
