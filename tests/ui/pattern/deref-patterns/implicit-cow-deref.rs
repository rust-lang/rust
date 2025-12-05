//@ run-pass
//! Test that implicit deref patterns interact as expected with `Cow` constructor patterns.
#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::borrow::Cow;
use std::rc::Rc;

fn main() {
    let cow: Cow<'static, [u8]> = Cow::Borrowed(&[1, 2, 3]);

    match cow {
        [..] => {}
    }

    match cow {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => unreachable!(),
    }

    match Rc::new(&cow) {
        Cow::Borrowed { 0: _ } => {}
        Cow::Owned { 0: _ } => unreachable!(),
    }

    let cow_of_cow: Cow<'_, Cow<'static, [u8]>> = Cow::Owned(cow);

    match cow_of_cow {
        [..] => {}
    }

    // This matches on the outer `Cow` (the owned one).
    match cow_of_cow {
        Cow::Borrowed(_) => unreachable!(),
        Cow::Owned(_) => {}
    }

    match Rc::new(&cow_of_cow) {
        Cow::Borrowed { 0: _ } => unreachable!(),
        Cow::Owned { 0: _ } => {}
    }
}
