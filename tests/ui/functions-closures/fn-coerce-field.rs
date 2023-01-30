// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616
#![allow(non_camel_case_types)]

struct r<F> where F: FnOnce() {
    field: F,
}

pub fn main() {
    fn f() {}
    let _i: r<fn()> = r {field: f as fn()};
}
