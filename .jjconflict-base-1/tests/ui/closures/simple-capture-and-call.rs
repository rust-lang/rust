//! regression test for issue #1895
//@ run-pass

fn main() {
    let x = 1_usize;
    let y = || x;
    let _z = y();
}
