//@ check-pass

struct Foo;
impl Foo {
    const N: usize = 4;
}

struct S([u8; Foo::N]);

fn main() {
    let _ = S([0; Foo::N]);
}
