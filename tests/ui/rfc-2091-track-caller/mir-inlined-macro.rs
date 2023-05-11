// run-pass
// revisions: default mir-opt
//[default] compile-flags: -Zinline-mir=no
//[mir-opt] compile-flags: -Zmir-opt-level=4

use std::panic::Location;

macro_rules! f {
    () => {
        Location::caller()
    };
}

#[inline(always)]
fn g() -> &'static Location<'static> {
    f!()
}

fn main() {
    let loc = g();
    assert_eq!(loc.line(), 16);
    assert_eq!(loc.column(), 5);
}
