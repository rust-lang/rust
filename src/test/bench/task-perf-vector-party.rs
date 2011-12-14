// Vectors are allocated in the Rust kernel's memory region, use of
// which requires some amount of synchronization. This test exercises
// that synchronization by spawning a number of tasks and then
// allocating and freeing vectors.

use std;
import vec;
import uint;
import str;
import task;

fn f(&&n: uint) {
    uint::range(0u, n) {|i|
        let v: [u8] = [];
        vec::reserve(v, 1000u);
    }
}

fn main(args: [str]) {
    let n =
        if vec::len(args) < 2u {
            100u
        } else { uint::parse_buf(str::bytes(args[1]), 10u) };
    uint::range(0u, 100u) {|i| task::spawn(copy n, f); }
}
