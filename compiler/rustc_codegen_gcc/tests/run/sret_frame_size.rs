// Compiler:
//
// Run-time:
//   status: 0

// A call returning in memory must get its destination as the return slot. A backend that returns
// into a temporary and then copies it to the destination needs two copies of the value in the
// caller's frame, which fails these assertions, for direct calls and for calls through pointers.

use std::hint::black_box;

const LEN: usize = 8192;
const BIG_SIZE: usize = LEN * size_of::<u64>();

type Big = [u64; LEN];

#[inline(never)]
fn make(seed: u64) -> Big {
    let mut array = [seed; LEN];
    array[1] = seed + 1;
    array
}

/// The size of this function's frame: the distance between `value` in two nested calls.
#[inline(never)]
fn direct_call_frame_size(depth: u32, outer_address: usize) -> usize {
    let value = make(depth.into());
    let address = black_box(&value) as *const Big as usize;
    let size = if depth == 0 {
        outer_address.abs_diff(address)
    } else {
        direct_call_frame_size(depth - 1, address)
    };
    // Keeps `value` alive across the recursive call.
    black_box(&value);
    size
}

/// The size of this function's frame: the distance between `value` in two nested calls.
#[inline(never)]
fn pointer_call_frame_size(depth: u32, outer_address: usize, make: fn(u64) -> Big) -> usize {
    let value = make(depth.into());
    let address = black_box(&value) as *const Big as usize;
    let size = if depth == 0 {
        outer_address.abs_diff(address)
    } else {
        pointer_call_frame_size(depth - 1, address, make)
    };
    // Keeps `value` alive across the recursive call.
    black_box(&value);
    size
}

fn main() {
    let limit = BIG_SIZE * 3 / 2;

    let size = direct_call_frame_size(black_box(1), 0);
    assert!(size < limit, "direct call: {}-byte frame for one {}-byte value", size, BIG_SIZE);

    let size = pointer_call_frame_size(black_box(1), 0, black_box(make));
    assert!(
        size < limit,
        "call through a pointer: {}-byte frame for one {}-byte value",
        size,
        BIG_SIZE
    );
}
