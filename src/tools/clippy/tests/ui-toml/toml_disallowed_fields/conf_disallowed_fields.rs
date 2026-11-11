#![warn(clippy::disallowed_fields)]
#![allow(clippy::match_single_binding)]

use std::ops::{Range, RangeTo};

struct X {
    y: u32,
}

enum Z {
    A { x: u32 },
    B { x: u32 },
}

use crate::X as Y;

fn b(X { y }: X) {}
//~^ disallowed_fields

fn main() {
    let x = X { y: 0 };
    let _ = x.y;
    //~^ disallowed_fields

    let x = Y { y: 0 };
    let _ = x.y;
    //~^ disallowed_fields

    let x = Range { start: 0, end: 0 };
    let _ = x.start;
    //~^ disallowed_fields
    let _ = x.end;
    //~^ disallowed_fields
    let Range { start, .. } = x;
    //~^ disallowed_fields

    let x = RangeTo { end: 0 };
    let _ = x.end;
    //~^ disallowed_fields

    match x {
        RangeTo { end } => {}, //~ disallowed_fields
    }

    let x = Z::B { x: 0 };
    match x {
        Z::A { x } => {},
        Z::B { x } => {}, //~ disallowed_fields
    }
}
