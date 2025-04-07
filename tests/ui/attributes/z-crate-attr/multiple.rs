//@ compile-flags: -Zcrate-attr=feature(foo),feature(bar)

fn main() {}

//~? ERROR invalid crate attribute
