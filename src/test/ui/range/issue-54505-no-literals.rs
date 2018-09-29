// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-rustfix

// Regression test for changes introduced while fixing #54505

// This test uses non-literals for Ranges
// (expecting no parens with borrow suggestion)

use std::ops::RangeBounds;


// take a reference to any built-in range
fn take_range(_r: &impl RangeBounds<i8>) {}


fn main() {
    take_range(std::ops::Range { start: 0, end: 1 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &std::ops::Range { start: 0, end: 1 }

    take_range(::std::ops::Range { start: 0, end: 1 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &::std::ops::Range { start: 0, end: 1 }

    take_range(std::ops::RangeFrom { start: 1 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &std::ops::RangeFrom { start: 1 }

    take_range(::std::ops::RangeFrom { start: 1 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &::std::ops::RangeFrom { start: 1 }

    take_range(std::ops::RangeFull {});
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &std::ops::RangeFull {}

    take_range(::std::ops::RangeFull {});
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &::std::ops::RangeFull {}

    take_range(std::ops::RangeInclusive::new(0, 1));
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &std::ops::RangeInclusive::new(0, 1)

    take_range(::std::ops::RangeInclusive::new(0, 1));
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &::std::ops::RangeInclusive::new(0, 1)

    take_range(std::ops::RangeTo { end: 5 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &std::ops::RangeTo { end: 5 }

    take_range(::std::ops::RangeTo { end: 5 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &::std::ops::RangeTo { end: 5 }

    take_range(std::ops::RangeToInclusive { end: 5 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &std::ops::RangeToInclusive { end: 5 }

    take_range(::std::ops::RangeToInclusive { end: 5 });
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &::std::ops::RangeToInclusive { end: 5 }
}
