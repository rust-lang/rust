// We should not suggest `self` in bare functions.
// And a note for RFC 1685 should not be shown.
// See #144968

//@ edition:2018

fn is_even(value) -> bool { //~ ERROR expected one of `:`, `@`, or `|`, found `)`
    value % 2 == 0
}

struct S;

impl S {
    fn is_even(value) -> bool { //~ ERROR expected one of `:`, `@`, or `|`, found `)`
        value % 2 == 0
    }
}

trait T {
    fn is_even(value) -> bool { //~ ERROR expected one of `:`, `@`, or `|`, found `)`
        value % 2 == 0
    }
}

fn main() {}
