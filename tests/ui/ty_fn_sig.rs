//@ check-pass
// Regression test

pub fn retry<F: Fn()>(f: F) {
    for _i in 0.. {
        f();
    }
}

fn main() {
    for y in 0..4 {
        let func = || ();
        func();
    }
}
