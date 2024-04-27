// Opaque type.

    #![feature(type_alias_impl_trait)]

pub type Adder<F, T>
where
    T: Clone,
    F: Copy
    = impl Fn(T) -> T;

pub type Adderrr<T> = impl Fn(  T  ) -> T;

impl Foo for Bar {
type E  = impl Trait;
}

pub type Adder_without_impl<F, T>
where
    T: Clone,
    F: Copy
    = Fn(T) -> T;

pub type Adderrr_without_impl<T> = Fn(  T  ) -> T;
