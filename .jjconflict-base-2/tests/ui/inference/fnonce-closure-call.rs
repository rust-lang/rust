//! Regression test for https://github.com/rust-lang/rust/issues/10718

//@ run-pass

fn f<F:FnOnce()>(p: F) {
    p();
}

pub fn main() {
    let p = || ();
    f(p);
}
