//! Regression test for <https://github.com/rust-lang/rust/issues/19380>.

trait Qiz {
  fn qiz();
}

struct Foo;
impl Qiz for Foo {
  fn qiz() {}
}

struct Bar {
  foos: &'static [&'static (dyn Qiz + 'static)]
//~^ ERROR E0038
}

const FOO : Foo = Foo;
const BAR : Bar = Bar { foos: &[&FOO]};
//~^ ERROR E0038

fn main() { }
