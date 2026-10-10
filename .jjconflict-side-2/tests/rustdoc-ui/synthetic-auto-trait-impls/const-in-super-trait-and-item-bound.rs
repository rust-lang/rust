// We used to ICE here while trying to synthesize auto trait impls.
// issue: 107715
//@ check-pass

pub const N: usize = 1;

pub struct MapType<K: Supertrait<V>, V> {
    _array: K::Array,
}

pub trait Subtrait: Supertrait<[u8; N]> {}

pub trait Supertrait<V> {
    type Array: AnotherTrait<V>;
}

pub trait AnotherTrait<V> {
    const LENGTH: usize;
}

pub struct Container<S: Subtrait> {
    _x: MapType<S, [u8; N]>,
}
