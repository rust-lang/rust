// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

trait Send {
    fn f(&self);
}

fn f<T:Send>(t: T) {
    t.f();
}

pub fn main() {
}
