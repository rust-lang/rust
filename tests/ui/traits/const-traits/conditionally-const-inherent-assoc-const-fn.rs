//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo(&self) {}
}

struct Bar<T>(T);

impl<T> Bar<T> {
    const fn foo(&self) where T: [const] Foo {
        self.0.foo()
    }
}

fn main() {}
