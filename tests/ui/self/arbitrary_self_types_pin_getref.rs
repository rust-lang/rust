// Confirms that Pin::get_ref can no longer shadow methods in pointees
// once arbitrary_self_types is enabled.
//
//@ revisions: default feature
#![cfg_attr(feature, feature(arbitrary_self_types))]

//@[default] check-pass

#![allow(dead_code)]

use std::pin::Pin;
use std::pin::pin;

struct A;

impl A {
    fn get_ref(self: &Pin<&A>) {}  // note &Pin
}

fn main() {
    let pinned_a: Pin<&mut A> = pin!(A);
    let pinned_a: Pin<&A> = pinned_a.as_ref();
    let _ = pinned_a.get_ref();
    //[feature]~^ ERROR: multiple applicable items
}
