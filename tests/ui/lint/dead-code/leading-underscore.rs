//@ run-pass

#![deny(dead_code)]

static _X: usize = 0;

fn _foo() {}

struct _Y {
    _z: usize,
}

enum _Z {}

impl _Y {
    fn _bar() {}
}

type _A = isize;

mod _bar {
    fn _qux() {}
}

extern "C" {
    #[link_name = "abort"]
    fn _abort() -> !;
}

pub fn main() {}
