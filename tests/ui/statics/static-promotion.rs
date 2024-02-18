//@ run-pass

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
static STR: &'static [u8] = b"hi";
static C: A<B<B<[u8]>>> = {
    A(&B {
        x: &B { x: STR },
    })
};

pub struct Slice(&'static [i32]);

static CONTENT: i32 = 42;
pub static CONTENT_MAP: Slice = Slice(&[CONTENT]);

pub static FOO: (i32, i32) = (42, 43);
pub static CONTENT_MAP2: Slice = Slice(&[FOO.0]);

fn main() {
    assert_eq!(b"hi", C.0.x.x);
    assert_eq!(&[42], CONTENT_MAP.0);
    assert_eq!(&[42], CONTENT_MAP2.0);
}
