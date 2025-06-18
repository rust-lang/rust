//@ compile-flags: -Z mir-opt-level=3
//@ run-pass

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

struct Baz<T> {
    a: T
}

fn main() {
    let d : Baz<[i32; 4]> = Baz { a: [1,2,3,4] };
    assert_eq!([1, 2, 3, 4], d.a);
}
