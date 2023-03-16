// check-pass
#![crate_type = "lib"]

const N: usize = 1;

trait Supertrait<V> {
    type AssociatedType;
}

trait Subtrait: Supertrait<[u8; N]> {}

struct MapType<K: Supertrait<V>, V> {
    map: K::AssociatedType,
}

struct Container<S: Subtrait> {
    _x: MapType<S, [u8; N]>,
}
