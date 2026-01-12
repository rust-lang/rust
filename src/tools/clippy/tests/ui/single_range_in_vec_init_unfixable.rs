//@no-rustfix
#![warn(clippy::single_range_in_vec_init)]

use std::ops::Range;

fn issue16306(v: &[i32]) {
    fn takes_range_slice(_: &[Range<i64>]) {}

    let len = v.len();
    takes_range_slice(&[0..len as i64]);
    //~^ single_range_in_vec_init
}
