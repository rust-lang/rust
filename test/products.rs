#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn tuple() -> (i64,) {
    (1,)
}

#[miri_run]
fn tuple_2() -> (i64, i64) {
    (1, 2)
}

#[miri_run]
fn tuple_5() -> (i64, i64, i64, i64, i64) {
    (1, 2, 3, 4, 5)
}

struct Pair { x: i64, y: i64 }

#[miri_run]
fn pair() -> Pair {
    Pair { x: 10, y: 20 }
}
