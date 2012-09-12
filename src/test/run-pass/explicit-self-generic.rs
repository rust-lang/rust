extern mod std;

/**
 * A function that returns a hash of a value
 *
 * The hash should concentrate entropy in the lower bits.
 */
type HashFn<K> = pure fn~(K) -> uint;
type EqFn<K> = pure fn~(K, K) -> bool;

enum LinearMap<K,V> {
    LinearMap_({
        resize_at: uint,
        size: uint})
}

fn linear_map<K,V>() -> LinearMap<K,V> {
    LinearMap_({
        resize_at: 32,
        size: 0})
}

impl<K,V> LinearMap<K,V> {
    fn len(&mut self) -> uint {
        self.size
    }
}

fn main() {
    let mut m = ~linear_map::<(),()>();
    assert m.len() == 0;
}

