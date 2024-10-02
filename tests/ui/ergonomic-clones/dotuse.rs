//@ check-pass

#![feature(ergonomic_clones)]

fn basic_test(x: i32) -> i32 {
    x.use.use.abs()
}

fn do_not_move_test(x: String) -> String {
    let s = x.use;
    x
}

fn main() {}
