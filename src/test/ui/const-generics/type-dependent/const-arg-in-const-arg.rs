// run-pass
#![feature(const_generics)]
#![allow(incomplete_features)]
#![feature(const_fn)]

struct Foo;

impl Foo {
    fn foo<const N: usize>(&self) -> usize {
        let f = self;
        f.bar::<{
            let f = Foo;
            f.bar::<7>()
        }>() + N
    }

    const fn bar<const M: usize>(&self) -> usize {
        M
    }
}

fn main() {
    let f = Foo;

    assert_eq!(f.foo::<13>(), 20)
}
