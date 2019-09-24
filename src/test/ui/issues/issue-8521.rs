// build-pass (FIXME(62277): could be check-pass?)
trait Foo1 {}

trait A {}

macro_rules! foo1(($t:path) => {
    impl<T: $t> Foo1 for T {}
});

foo1!(A);

trait Foo2 {}

trait B<T> {}

#[allow(unused)]
struct C {}

macro_rules! foo2(($t:path) => {
    impl<T: $t> Foo2 for T {}
});

foo2!(B<C>);

fn main() {}
