//@no-rustfix
#![warn(clippy::single_range_in_vec_init)]

use std::ops::Range;

fn issue16306(v: &[i32]) {
    fn takes_range_slice(_: &[Range<i64>]) {}

    let len = v.len();
    takes_range_slice(&[0..len as i64]);
    //~^ single_range_in_vec_init
}

#[allow(clippy::no_effect, clippy::useless_vec)]
fn issue16508_open_ended() {
    [..10];
    //~^ single_range_in_vec_init
    vec![..10];
    //~^ single_range_in_vec_init

    [10..];
    //~^ single_range_in_vec_init
    vec![10..];
    //~^ single_range_in_vec_init

    [..=10];
    //~^ single_range_in_vec_init
    vec![..=10];
    //~^ single_range_in_vec_init

    [..];
    //~^ single_range_in_vec_init
    vec![..];
    //~^ single_range_in_vec_init
}
