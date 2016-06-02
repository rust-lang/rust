#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

//error-pattern:static should have been cached


#[miri_run]
fn failed_assertions() {
    assert_eq!(5, 6);
}

fn main() {}
