// run-pass
// pretty-expanded FIXME #23616

fn f<F:FnOnce()>(p: F) {
    p();
}

pub fn main() {
    let p = || ();
    f(p);
}
