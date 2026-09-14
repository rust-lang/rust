#![feature(const_trait_impl, const_convert)]

//@ check-pass

const trait Convert<T> {
    fn to(self) -> T;
}

const impl<A, B> Convert<B> for A
where
    B: [const] From<A>,
{
    fn to(self) -> B {
        B::from(self)
    }
}

const FOO: fn() -> String = || "foo".to();

fn main() {}
