// Existential type.

    #![feature(existential_type)]

pub existential type Adder<F, T>
where
    T: Clone,
    F: Copy
    : Fn(T) -> T;

pub existential type Adderrr<T>: Fn(  T  ) -> T;

impl Foo for Bar {
existential type E  : Trait;
}
