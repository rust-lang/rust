#![allow(unused)]
#![warn(clippy::ref_option_ref)]

type OptRefU32<'a> = &'a Option<&'a u32>;
type OptRef<'a, T> = &'a Option<&'a T>;

fn main() {
    let x: &Option<&u32> = &None;
}
