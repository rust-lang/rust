// known-bug: #110395
// FIXME check-pass
#![feature(const_trait_impl, effects)]

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
