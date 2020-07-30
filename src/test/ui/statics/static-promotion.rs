// check-pass

// Use of global static variables in literal values should be allowed for
// promotion.
// This test is to demonstrate the issue raised in
// https://github.com/rust-lang/rust/issues/70584

// Literal values were previously promoted into local static values when
// other global static variables are used.

struct A<T: 'static>(&'static T);
struct B<T: 'static + ?Sized> {
    x: &'static T,
}
static C: A<B<B<[u8]>>> = {
    A(&B {
        x: &B { x: b"hi" as &[u8] },
    })
};

fn main() {
    assert_eq!(b"hi", C.0.x.x);
}
