
use std;
import std::ivec;

fn main() {
    let v: [mutable int] = ~[mutable 10, 20];
    assert (v.(0) == 10);
    assert (v.(1) == 20);
    ivec::reverse(v);
    assert (v.(0) == 20);
    assert (v.(1) == 10);
    let v2 = ivec::reversed[int](~[10, 20]);
    assert (v2.(0) == 20);
    assert (v2.(1) == 10);
    v.(0) = 30;
    assert (v2.(0) == 20);
    // Make sure they work with 0-length vectors too.

    let v4 = ivec::reversed[int](~[]);
    let v3: [mutable int] = ~[mutable];
    ivec::reverse[int](v3);
}