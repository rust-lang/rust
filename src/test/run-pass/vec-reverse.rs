
use std;
import std::vec;

fn main() {
    let vec[mutable int] v = [mutable 10, 20];
    assert (v.(0) == 10);
    assert (v.(1) == 20);
    vec::reverse(v);
    assert (v.(0) == 20);
    assert (v.(1) == 10);
    auto v2 = vec::reversed[int]([10, 20]);
    assert (v2.(0) == 20);
    assert (v2.(1) == 10);
    v.(0) = 30;
    assert (v2.(0) == 20);
    // Make sure they work with 0-length vectors too.

    auto v4 = vec::reversed[int]([]);
    let vec[mutable int] v3 = vec::empty_mut();
    vec::reverse[int](v3);
}