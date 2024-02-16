//@ run-pass
#![allow(stable_features)]
#![allow(unused_labels)]
#![allow(unreachable_code)]

macro_rules! x {
    ($a:lifetime) => {
        $a: loop {
            break $a;
            panic!("failed");
        }
    }
}
macro_rules! br {
    ($a:lifetime) => {
        break $a;
    }
}
macro_rules! br2 {
    ($b:lifetime) => {
        'b: loop {
            break $b; // this $b should refer to the outer loop.
        }
    }
}
fn main() {
    x!('a);
    'c: loop {
        br!('c);
        panic!("failed");
    }
    'b: loop {
        br2!('b);
        panic!("failed");
    }
}
