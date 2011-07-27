
use std;
import std::box;

#[test]
fn test() {
    let x = @3;
    let y = @3;
    assert (box::ptr_eq[int](x, x));
    assert (box::ptr_eq[int](y, y));
    assert (!box::ptr_eq[int](x, y));
    assert (!box::ptr_eq[int](y, x));
}