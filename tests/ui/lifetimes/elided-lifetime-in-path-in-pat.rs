//@ check-pass

struct Foo<'a> {
    x: &'a (),
}

// The lifetime in pattern-position `Foo` is elided.
// Verify that lowering does not create an independent lifetime parameter for it.
fn foo<'a>(Foo { x }: Foo<'a>) {
    *x
}

fn main() {}
