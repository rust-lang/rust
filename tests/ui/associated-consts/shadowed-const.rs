// Checking that none of these ICE, which was introduced in
// https://github.com/rust-lang/rust/issues/93553
trait Foo {
    type Bar;
}

trait Baz: Foo {
    const Bar: Self::Bar;
}

trait Baz2: Foo {
    const Bar: u32;

    fn foo() -> Self::Bar;
}

trait Baz3 {
  const BAR: usize;
  const QUX: Self::BAR;
  //~^ ERROR found associated const
}

fn main() {}
