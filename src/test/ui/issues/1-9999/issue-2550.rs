// run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

// pretty-expanded FIXME #23616

struct C {
    x: usize,
}

fn C(x: usize) -> C {
    C {
        x: x
    }
}

fn f<T>(_x: T) {
}

pub fn main() {
    f(C(1));
}
