

// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
import std::sys::rustrt::size_of;
use std;

fn main() {
    assert (size_of[u8]() == 1 as uint);
    assert (size_of[u32]() == 4 as uint);
    assert (size_of[char]() == 4 as uint);
    assert (size_of[i8]() == 1 as uint);
    assert (size_of[i32]() == 4 as uint);
    assert (size_of[{a: u8, b: i8}]() == 2 as uint);
    assert (size_of[{a: u8, b: i8, c: u8}]() == 3 as uint);
    // Alignment causes padding before the char and the u32.

    assert (size_of[{a: u8, b: i8, c: {u: char, v: u8}, d: u32}]() ==
                16 as uint);
    assert (size_of[int]() == size_of[uint]());
    assert (size_of[{a: int, b: ()}]() == size_of[int]());
    assert (size_of[{a: int, b: (), c: ()}]() == size_of[int]());
    assert (size_of[int]() == size_of[{x: int}]());
}