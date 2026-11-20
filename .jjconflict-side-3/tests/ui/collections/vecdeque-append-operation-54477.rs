// https://github.com/rust-lang/rust/issues/54477
//@ run-pass
// rust-lang/rust#54477: runtime bug in the VecDeque library that was
// exposed by this test case, derived from test suite of crates.io
// `collection` crate.

use std::collections::VecDeque;

fn main() {
    let mut vecdeque_13 = VecDeque::from(vec![ ]);
    let mut vecdeque_29 = VecDeque::from(vec![ 0 ]);
    vecdeque_29.insert(0,  30 );
    vecdeque_29.insert(1,  31 );
    vecdeque_29.insert(2,  32 );
    vecdeque_29.insert(3,  33 );
    vecdeque_29.insert(4,  34 );
    vecdeque_29.insert(5,  35 );
    // println!("vecdeque_13: {:?}", vecdeque_13);
    // println!("vecdeque_29: {:?}", vecdeque_29);

    // println!("Invoking: `vecdeque_13.append(&mut vecdeque_29)`");
    vecdeque_13.append(&mut vecdeque_29);

    // println!("vecdeque_13: {:?}", vecdeque_13);

    assert_eq!(vecdeque_13, VecDeque::from(vec![30, 31, 32, 33, 34, 35, 0]));
}
