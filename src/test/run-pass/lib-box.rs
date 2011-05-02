use std;
import std.Box;

fn main() {
    auto x = @3;
    auto y = @3;
    assert (Box.ptr_eq[int](x, x));
    assert (Box.ptr_eq[int](y, y));
    assert (!Box.ptr_eq[int](x, y));
    assert (!Box.ptr_eq[int](y, x));
}

