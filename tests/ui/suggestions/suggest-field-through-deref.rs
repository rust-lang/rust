// run-rustfix
#![allow(dead_code)]
use std::sync::Arc;
struct S {
    long_name: (),
    foo: (),
}
fn main() {
    let x = Arc::new(S { long_name: (), foo: () });
    let _ = x.longname; //~ ERROR no field `longname`
    let y = S { long_name: (), foo: () };
    let _ = y.longname; //~ ERROR no field `longname`
}
