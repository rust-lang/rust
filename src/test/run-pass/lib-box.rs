
use std;
import std::box;

fn main() {
    auto x = @3;
    auto y = @3;
    assert (box::ptr_eq[int](x, x));
    assert (box::ptr_eq[int](y, y));
    assert (!box::ptr_eq[int](x, y));
    assert (!box::ptr_eq[int](y, x));
}