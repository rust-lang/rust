
use std;
import std::vec;

fn main() {
    let v: vec[mutable int] = [mutable 10, 20];
    assert (v.(0) == 10);
    assert (v.(1) == 20);
    vec::reverse(v);
    assert (v.(0) == 20);
    assert (v.(1) == 10);
    let v2 = vec::reversed[int]([10, 20]);
    assert (v2.(0) == 20);
    assert (v2.(1) == 10);
    v.(0) = 30;
    assert (v2.(0) == 20);
    // Make sure they work with 0-length vectors too.

    let v4 = vec::reversed[int]([]);
    let v3: vec[mutable int] = vec::empty_mut();
    vec::reverse[int](v3);
}