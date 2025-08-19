//@ check-pass

#![feature(const_trait_impl)]

#[const_trait]
trait A where Self::Assoc: const B {
    type Assoc;
}

#[const_trait]
trait B {}

fn needs_b<T: const B>() {}

fn test<T: A>() {
    needs_b::<T::Assoc>();
}

fn main() {}
