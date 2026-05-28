//@ run-rustfix
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
    let a = Some(Arc::new(S { long_name: (), foo: () }));
    let _ = a.longname; //~ ERROR no field `longname`
    let b = Some(S { long_name: (), foo: () });
    let _ = b.long_name; //~ ERROR no field `long_name`
    let c = Ok::<_, ()>(Arc::new(S { long_name: (), foo: () }));
    let _ = c.longname; //~ ERROR no field `longname`
    let d = Ok::<_, ()>(S { long_name: (), foo: () });
    let _ = d.long_name; //~ ERROR no field `long_name`
}
