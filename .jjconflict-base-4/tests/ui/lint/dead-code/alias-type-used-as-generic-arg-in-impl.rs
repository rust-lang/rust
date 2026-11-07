//@ check-pass

#![deny(dead_code)]

struct T<X>(X);

type A<X> = T<X>;

trait Tr {
    fn foo();
}

impl<X> Tr for T<A<X>> {
    fn foo() {}
}

fn main() {
   T::<T<()>>::foo();
}
