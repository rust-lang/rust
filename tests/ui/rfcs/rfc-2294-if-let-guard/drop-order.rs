// check drop order of temporaries create in match guards.
// For normal guards all temporaries are dropped before the body of the arm.
// For let guards temporaries live until the end of the arm.

//@ run-pass

#![feature(if_let_guard)]
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

fn if_guard(num: i32) {
    let _d = make_d(1);
    match num {
        1 | 2 if make_d(2).0 == 2 => {
            make_d(3);
        }
        _ => {}
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

fn main() {
    if_guard(1);
    if_guard(2);
    if_let_guard(1);
    if_let_guard(2);
    let expected =  [
        1, 2, !2, 3, !3, !1,
        1, 2, !2, 3, !3, !1,
        1, 2, 3, !3, !2, !1,
        1, 2, 3, !3, !2, !1,
    ];
    assert_eq!(*A.lock().unwrap(), expected);
}
