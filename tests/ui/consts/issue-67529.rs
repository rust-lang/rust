//@ compile-flags: -Z mir-opt-level=3
//@ run-pass

struct Baz<T: ?Sized> {
    a: T
}

fn main() {
    let d : Baz<[i32; 4]> = Baz { a: [1,2,3,4] };
    assert_eq!([1, 2, 3, 4], d.a);
}
