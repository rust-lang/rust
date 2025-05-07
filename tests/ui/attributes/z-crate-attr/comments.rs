//@ check-pass
//@ compile-flags: -Zcrate-attr=/*hi-there*/feature(rustc_attrs)

#[rustc_dummy]
fn main() {}
