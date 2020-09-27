#![warn(clippy::needless_range_loop, clippy::manual_memcpy)]

pub fn manual_copy_with_counters(src: &[i32], dst: &mut [i32], dst2: &mut [i32]) {
    let mut count = 0;
    for i in 3..src.len() {
        dst[i] = src[count];
        count += 1;
    }

    let mut count = 0;
    for i in 3..src.len() {
        dst[count] = src[i];
        count += 1;
    }

    let mut count = 3;
    for i in 0..src.len() {
        dst[count] = src[i];
        count += 1;
    }

    let mut count = 3;
    for i in 0..src.len() {
        dst[i] = src[count];
        count += 1;
    }

    let mut count = 0;
    for i in 3..(3 + src.len()) {
        dst[i] = src[count];
        count += 1;
    }

    let mut count = 3;
    for i in 5..src.len() {
        dst[i] = src[count - 2];
        count += 1;
    }

    let mut count = 5;
    for i in 3..10 {
        dst[i] = src[count];
        count += 1;
    }

    let mut count = 3;
    let mut count2 = 30;
    for i in 0..src.len() {
        dst[count] = src[i];
        dst2[count2] = src[i];
        count += 1;
        count2 += 1;
    }

    // make sure parentheses are added properly to bitwise operators, which have lower precedence than
    // arithmetric ones
    let mut count = 0 << 1;
    for i in 0..1 << 1 {
        dst[count] = src[i + 2];
        count += 1;
    }
}

fn main() {}
