//@ run-pass

fn f<F:FnOnce()>(p: F) {
    p();
}

pub fn main() {
    let p = || ();
    f(p);
}
