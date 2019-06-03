#![feature(core_intrinsics)]

use std::intrinsics::type_name;
use std::mem::{size_of, size_of_val};

fn main() {
    assert_eq!(size_of::<Option<i32>>(), 8);
    assert_eq!(size_of_val(&()), 0);
    assert_eq!(size_of_val(&42), 4);
    assert_eq!(size_of_val(&[] as &[i32]), 0);
    assert_eq!(size_of_val(&[1, 2, 3] as &[i32]), 12);
    assert_eq!(size_of_val("foobar"), 6);

    assert_eq!(unsafe { type_name::<Option<i32>>() }, "core::option::Option<i32>");
}
