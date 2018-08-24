// pretty-expanded FIXME #23616

struct r<F> where F: FnOnce() {
    field: F,
}

pub fn main() {
    fn f() {}
    let _i: r<fn()> = r {field: f as fn()};
}
