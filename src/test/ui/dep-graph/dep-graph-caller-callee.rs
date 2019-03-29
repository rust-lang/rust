// Test that immediate callers have to change when callee changes, but
// not callers' callers.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]

fn main() { }

mod x {
    #[rustc_if_this_changed]
    pub fn x() { }
}

mod y {
    use x;

    // These dependencies SHOULD exist:
    #[rustc_then_this_would_need(typeck_tables_of)] //~ ERROR OK
    pub fn y() {
        x::x();
    }
}

mod z {
    use y;

    // These are expected to yield errors, because changes to `x`
    // affect the BODY of `y`, but not its signature.
    #[rustc_then_this_would_need(typeck_tables_of)] //~ ERROR no path
    pub fn z() {
        y::y();
    }
}
