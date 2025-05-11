// Regression test for <https://github.com/rust-lang/rust/issues/137288>.

trait B {
    type C;
}

impl<U> B for &Missing {
//~^ ERROR cannot find type `Missing` in this scope
    type C = ();
}

struct E<T: B> {
    g: <T as B>::C,
}

fn h(i: Box<E<&()>>) {}

fn main() {}
