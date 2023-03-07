// check-pass
#![feature(const_trait_impl)]

#[const_trait]
trait Foo {
    fn foo(&self) {}
}

struct Bar<T>(T);

impl<T: ~const Foo> Bar<T> {
    const fn foo(&self) {
        self.0.foo()
    }
}

fn main() {}
