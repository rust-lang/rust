//@ check-pass
//@ compile-flags: -L native=does-not-exist -Link-everything-statically

fn main() {}

//~? WARN search path `does-not-exist` does not exist
//~? WARN search path `ink-everything-statically` does not exist
