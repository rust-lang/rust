#![warn(clippy::needless_range_loop, clippy::manual_memcpy)]
//@no-rustfix
pub fn manual_copy_with_counters(src: &[i32], dst: &mut [i32], dst2: &mut [i32]) {
    let mut count = 0;
    for i in 3..src.len() {
        //~^ manual_memcpy

        dst[i] = src[count];
        count += 1;
    }

    let mut count = 0;
    for i in 3..src.len() {
        //~^ manual_memcpy

        dst[count] = src[i];
        count += 1;
    }

    let mut count = 3;
    for i in 0..src.len() {
        //~^ manual_memcpy

        dst[count] = src[i];
        count += 1;
    }

    let mut count = 3;
    for i in 0..src.len() {
        //~^ manual_memcpy

        dst[i] = src[count];
        count += 1;
    }

    let mut count = 0;
    for i in 3..(3 + src.len()) {
        //~^ manual_memcpy

        dst[i] = src[count];
        count += 1;
    }

    let mut count = 3;
    for i in 5..src.len() {
        //~^ manual_memcpy

        dst[i] = src[count - 2];
        count += 1;
    }

    let mut count = 2;
    for i in 0..dst.len() {
        //~^ manual_memcpy

        dst[i] = src[count];
        count += 1;
    }

    let mut count = 5;
    for i in 3..10 {
        //~^ manual_memcpy

        dst[i] = src[count];
        count += 1;
    }

    let mut count = 3;
    let mut count2 = 30;
    for i in 0..src.len() {
        //~^ manual_memcpy

        dst[count] = src[i];
        dst2[count2] = src[i];
        count += 1;
        count2 += 1;
    }

    // make sure parentheses are added properly to bitwise operators, which have lower precedence than
    // arithmetic ones
    let mut count = 0 << 1;
    for i in 0..1 << 1 {
        //~^ manual_memcpy

        dst[count] = src[i + 2];
        count += 1;
    }

    // make sure incrementing expressions without semicolons at the end of loops are handled correctly.
    let mut count = 0;
    for i in 3..src.len() {
        //~^ manual_memcpy

        dst[i] = src[count];
        count += 1
    }

    // make sure ones where the increment is not at the end of the loop.
    // As a possible enhancement, one could adjust the offset in the suggestion according to
    // the position. For example, if the increment is at the top of the loop;
    // treating the loop counter as if it were initialized 1 greater than the original value.
    let mut count = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..src.len() {
        count += 1;
        dst[i] = src[count];
    }
}

fn main() {}
