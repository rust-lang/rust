pub trait TraitB {
    type Item;
}

pub trait TraitA<A> {
    type Type;

    fn bar<T>(_: T) -> Self;

    fn baz<T>(_: T) -> Self
    where
        T: TraitB,
        <T as TraitB>::Item: Copy;

    const A: usize;
}
