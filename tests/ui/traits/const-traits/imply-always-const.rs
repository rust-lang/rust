//@ check-pass

#![feature(const_trait_impl)]

const trait A where Self::Assoc: const B {
    type Assoc;
}

const trait B {}

fn needs_b<T: const B>() {}

fn test<T: A>() {
    needs_b::<T::Assoc>();
}

fn main() {}
