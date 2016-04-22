#![feature(custom_attribute, box_syntax)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn make_box() -> Box<(i16, i16)> {
    Box::new((1, 2))
}

#[miri_run]
fn make_box_syntax() -> Box<(i16, i16)> {
    box (1, 2)
}

#[miri_run]
fn main() {
    assert_eq!(*make_box(), (1, 2));
    assert_eq!(*make_box_syntax(), (1, 2));
}
