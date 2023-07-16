// Regression test for issue #113736.
// check-pass

#![feature(lazy_type_alias)]

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
