// run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![allow(incomplete_features)]

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
