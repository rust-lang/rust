// xfail-test
type IMap<K: Copy, V: Copy> = ~[(K, V)];

trait ImmutableMap<K: Copy, V: Copy>
{
    pure fn contains_key(key: K) -> bool;
}

impl<K: Copy, V: Copy> IMap<K, V> : ImmutableMap<K, V>
{
    pure fn contains_key(key: K) -> bool
    {
        vec::find(self, |e| {e.first() == key}).is_some()
    }
}

fn main() {}