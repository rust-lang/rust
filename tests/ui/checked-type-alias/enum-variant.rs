// Regression test for issue #113736.
//@ check-pass

#![feature(checked_type_aliases)]

enum Enum {
    Unit,
    Tuple(),
    Struct {},
}

fn main() {
    type Alias = Enum;
    let _ = Alias::Unit;
    let _ = Alias::Tuple();
    let _ = Alias::Struct {};
}
