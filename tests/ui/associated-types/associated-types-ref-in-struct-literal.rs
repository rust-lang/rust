//@ run-pass
// Test associated type references in a struct literal. Issue #20535.


pub trait Foo {
    type Bar;

    fn dummy(&self) { }
}

impl Foo for isize {
    type Bar = isize;
}

struct Thing<F: Foo> {
    a: F,
    b: F::Bar,
}

fn main() {
    let thing = Thing{a: 1, b: 2};
    assert_eq!(thing.a + 1, thing.b);
}
