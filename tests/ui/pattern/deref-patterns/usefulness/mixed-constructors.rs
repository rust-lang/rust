//! Test matches with a mix of ADT constructors and deref patterns. Currently, usefulness analysis
//! doesn't support this, so make sure we catch it beforehand. As a consequence, it takes priority
//! over non-exhaustive match and unreachable pattern errors.
#![feature(deref_patterns)]
#![expect(incomplete_features)]
#![deny(unreachable_patterns)]

use std::borrow::Cow;

fn main() {
    let cow: Cow<'static, bool> = Cow::Borrowed(&false);

    match cow {
        true => {}
        //~v ERROR mix of deref patterns and normal constructors
        false => {}
        Cow::Borrowed(_) => {}
    }

    match cow {
        Cow::Owned(_) => {}
        Cow::Borrowed(_) => {}
        true => {}
        //~^ ERROR mix of deref patterns and normal constructors
    }

    match cow {
        _ => {}
        Cow::Owned(_) => {}
        false => {}
        //~^ ERROR mix of deref patterns and normal constructors
    }

    match (cow, 0) {
        (Cow::Owned(_), 0) => {}
        (Cow::Borrowed(_), 0) => {}
        (true, 0) => {}
        //~^ ERROR mix of deref patterns and normal constructors
    }

    match (0, cow) {
        (0, Cow::Owned(_)) => {}
        (0, Cow::Borrowed(_)) => {}
        _ => {}
        (1, true) => {}
        //~^ ERROR mix of deref patterns and normal constructors
    }
}
