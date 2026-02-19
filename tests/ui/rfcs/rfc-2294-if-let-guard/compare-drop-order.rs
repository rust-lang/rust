//@ run-pass
//@revisions: edition2015 edition2018 edition2021 edition2024
//@[edition2015] edition:2015
//@[edition2018] edition:2018
//@[edition2021] edition:2021
//@[edition2024] edition:2024

#![allow(irrefutable_let_patterns)]

use std::sync::Mutex;

static A: Mutex<Vec<i32>> = Mutex::new(Vec::new());

struct D(i32);

fn make_d(x: i32) -> D {
    A.lock().unwrap().push(x);
    D(x)
}

impl Drop for D {
    fn drop(&mut self) {
        A.lock().unwrap().push(!self.0);
    }
}

fn if_let_guard(num: i32) {
    let _d = make_d(1);
    match num {
        1 | 2 if let D(ref _x) = make_d(2) => {
            make_d(3);
        }
        _ => {}
    }
}

fn if_let(num: i32) {
    let _d = make_d(1);
    match num {
        1 | 2 => {
            if let D(ref _x) = make_d(2)  {
                make_d(3);
            }
        }
        _ => {}
    }
}

fn main() {
    if_let(1);
    if_let(2);
    if_let_guard(1);
    if_let_guard(2);
    let expected =
        [1, 2, 3, !3, !2, !1, 1, 2, 3, !3, !2, !1,
        // Here is two parts, first one is for basic if let inside the match arm
        // And second part is for if let guard
         1, 2, 3, !3, !2, !1, 1, 2, 3, !3, !2, !1];
    assert_eq!(*A.lock().unwrap(), expected);
}
