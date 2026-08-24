// issue-link: https://github.com/rust-lang/rust/issues/160591
// A method call whose where-clause fails shouldn't ICE when checking the fn sig.

trait Context {}
struct Foo;
impl Foo {
    fn take<T: Context>(&self, _: T) {}
}

fn main() {
    let f = Foo;
    f.take(());
    //~^ ERROR the trait bound `(): Context` is not satisfied
}
