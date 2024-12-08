//@ run-rustfix
#![allow(unused)] // for the fixed file

trait Trait<T = Self> {
    type A;

    fn func(&self) -> Self::A;
}

struct S<T>(T);
impl<K> S<K> {
    fn foo<'a, T: Trait + 'a>(&self, _: impl Trait, x: impl Trait, _: T) {
        qux(x.func()) //~ ERROR mismatched types
    }

    fn ban<T>(x: T) where T: Trait {
        qux(x.func()) //~ ERROR mismatched types
    }
}

fn foo<'a, T: Trait + 'a>(_: impl Trait, x: impl Trait, _: T) {
    qux(x.func()) //~ ERROR mismatched types
}

fn bar<T: Trait>(x: T) {
    qux(x.func()) //~ ERROR mismatched types
}

fn foo2(x: impl Trait<i32>) {
    qux(x.func()) //~ ERROR mismatched types
}

fn bar2<T: Trait<i32>>(x: T) {
    qux(x.func()) //~ ERROR mismatched types
}

fn ban<T>(x: T) where T: Trait {
    qux(x.func()) //~ ERROR mismatched types
}

fn qux(_: usize) {}

fn main() {}
