// Check that `rustfmt` does not remove view type syntax.

struct Foo {
    bar: u8,
}

impl Foo {
    fn a(&mut self.{ bar }) {}
    fn c(&mut self.{}) {}

    fn d(a: &mut Self.{ bar }) {}
    fn d(a: &mut Self.{}) {}
}
