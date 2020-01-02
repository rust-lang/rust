// Tests for miscompilation due to const propagation, as described in #67529. This used to result
// in an assertion error, because d.a was replaced with a new allocation that was never
// initialized.
//
// run-pass
// compile-flags: -Zmir-opt-level=2
struct Baz<T: ?Sized> {
    a: T,
}

fn main() {
    let d: Baz<[i32; 4]> = Baz { a: [1, 2, 3, 4] };
    assert_eq!([1, 2, 3, 4], d.a);
}
