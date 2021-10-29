// run-rustfix
// edition:2018

#![feature(custom_inner_attributes)]
#![warn(clippy::needless_splitn)]
#![allow(clippy::iter_skip_next, clippy::iter_nth_zero, clippy::manual_split_once)]

extern crate itertools;

#[allow(unused_imports)]
use itertools::Itertools;

fn main() {
    let str = "key=value=end";
    let _ = str.splitn(2, '=').next();
    let _ = str.splitn(2, '=').nth(0);
    let _ = str.splitn(2, '=').nth(1);
    let (_, _) = str.splitn(2, '=').next_tuple().unwrap();
    let (_, _) = str.splitn(3, '=').next_tuple().unwrap();
    let _: Vec<&str> = str.splitn(3, '=').collect();

    let _ = str.rsplitn(2, '=').next();
    let _ = str.rsplitn(2, '=').nth(0);
    let _ = str.rsplitn(2, '=').nth(1);
    let (_, _) = str.rsplitn(2, '=').next_tuple().unwrap();
    let (_, _) = str.rsplitn(3, '=').next_tuple().unwrap();
}
