//@ run-pass
#![allow(dead_code)]

trait Send {
    fn f(&self);
}

fn f<T:Send>(t: T) {
    t.f();
}

pub fn main() {
}
