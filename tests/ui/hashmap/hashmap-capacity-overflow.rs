//@ run-fail
//@ error-pattern:capacity overflow
//@ ignore-emscripten no processes

use std::collections::hash_map::HashMap;
use std::mem::size_of;

fn main() {
    let threshold = usize::MAX / size_of::<(u64, u64, u64)>();
    let mut h = HashMap::<u64, u64>::with_capacity(threshold + 100);
    h.insert(0, 0);
}
