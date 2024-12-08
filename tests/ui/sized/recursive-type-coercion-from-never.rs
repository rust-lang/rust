//@ build-fail
//~^ ERROR cycle detected when computing layout of `Foo<()>`

// Regression test for a stack overflow: https://github.com/rust-lang/rust/issues/113197

trait A { type Assoc; }

impl A for () {
    type Assoc = Foo<()>;
}

struct Foo<T: A>(T::Assoc);

fn main() {
    Foo::<()>(todo!());
}
