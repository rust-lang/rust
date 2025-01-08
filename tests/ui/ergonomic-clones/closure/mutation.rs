//@ known-bug: unknown
// This test currently ICEs, need fix

#![feature(ergonomic_clones)]

fn main() {
    let mut my_var = false;
    let callback = use || {
        my_var = true;
    };
    callback();
}
