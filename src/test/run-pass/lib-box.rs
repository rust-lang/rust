// xfail-stage0

use std;
import std.Box;

fn main() {
    auto x = @3;
    auto y = @3;
    check (Box.ptr_eq[int](x, x));
    check (Box.ptr_eq[int](y, y));
    check (!Box.ptr_eq[int](x, y));
    check (!Box.ptr_eq[int](y, x));
}

