//@ run-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    let mut my_var = false;
    let mut callback = || {
        let _ = my_var.use;
        my_var = true;
    };
    callback();
    assert!(my_var);
}
