// run-pass
// Issue #922
// pretty-expanded FIXME #23616

fn f2<F>(_thing: F) where F: FnOnce() { }

fn f<F>(thing: F) where F: FnOnce() {
    f2(thing);
}

pub fn main() {
    f(|| {});
}
