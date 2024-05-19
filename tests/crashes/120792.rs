//@ known-bug: #120792
//@ compile-flags: -Zpolymorphize=on -Zinline-mir=yes

impl Trait<()> for () {
    fn foo<'a, K>(self, _: (), _: K) {
        todo!();
    }
}

trait Foo<T> {}

impl<F, T> Foo<T> for F {
    fn main() {
        ().foo((), ());
    }
}

trait Trait<T> {
    fn foo<'a, K>(self, _: T, _: K)
    where
        T: 'a,
        K: 'a;
}

pub fn main() {}
