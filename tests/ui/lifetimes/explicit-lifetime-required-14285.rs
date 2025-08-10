//! Regression test for https://github.com/rust-lang/rust/issues/14285

trait Foo {
    fn dummy(&self) { }
}

struct A;

impl Foo for A {}

struct B<'a>(&'a (dyn Foo + 'a));

fn foo<'a>(a: &dyn Foo) -> B<'a> {
    B(a)    //~ ERROR explicit lifetime required in the type of `a` [E0621]
}

fn main() {
    let _test = foo(&A);
}
