use std::fmt::Display;
use std::ops::Deref;

trait Foo {
    fn bar(self) -> impl Deref<Target = impl Display + ?Sized>;
}

fn foo<T: Foo>(t: T) {
    let () = t.bar();
    //~^ ERROR mismatched types
}

fn main() {}
