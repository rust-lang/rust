/*!
 * A simple map based on a vector for small integer keys. Space requirements
 * are O(highest integer key).
 */
#[forbid(deprecated_mode)];

use core::option;
use core::option::{Some, None};
use dvec::DVec;
use map::Map;

// FIXME (#2347): Should not be @; there's a bug somewhere in rustc that
// requires this to be.
type SmallIntMap_<T: Copy> = {v: DVec<Option<T>>};

pub enum SmallIntMap<T:Copy> {
    SmallIntMap_(@SmallIntMap_<T>)
}

/// Create a smallintmap
pub fn mk<T: Copy>() -> SmallIntMap<T> {
    let v = DVec();
    return SmallIntMap_(@{v: move v});
}

/**
 * Add a value to the map. If the map already contains a value for
 * the specified key then the original value is replaced.
 */
#[inline(always)]
pub fn insert<T: Copy>(self: SmallIntMap<T>, key: uint, val: T) {
    //io::println(fmt!("%?", key));
    self.v.grow_set_elt(key, &None, Some(val));
}

/**
 * Get the value for the specified key. If the key does not exist
 * in the map then returns none
 */
pub pure fn find<T: Copy>(self: SmallIntMap<T>, key: uint) -> Option<T> {
    if key < self.v.len() { return self.v.get_elt(key); }
    return None::<T>;
}

/**
 * Get the value for the specified key
 *
 * # Failure
 *
 * If the key does not exist in the map
 */
pub pure fn get<T: Copy>(self: SmallIntMap<T>, key: uint) -> T {
    match find(self, key) {
      None => {
        error!("smallintmap::get(): key not present");
        fail;
      }
      Some(move v) => return v
    }
}

/// Returns true if the map contains a value for the specified key
pub pure fn contains_key<T: Copy>(self: SmallIntMap<T>, key: uint) -> bool {
    return !find(self, key).is_none();
}

/// Implements the map::map interface for smallintmap
impl<V: Copy> SmallIntMap<V>: map::Map<uint, V> {
    pure fn size() -> uint {
        let mut sz = 0u;
        for self.v.each |item| {
            match *item {
              Some(_) => sz += 1u,
              _ => ()
            }
        }
        sz
    }
    #[inline(always)]
    fn insert(key: uint, value: V) -> bool {
        let exists = contains_key(self, key);
        insert(self, key, value);
        return !exists;
    }
    fn remove(key: uint) -> bool {
        if key >= self.v.len() {
            return false;
        }
        let old = self.v.get_elt(key);
        self.v.set_elt(key, None);
        old.is_some()
    }
    fn clear() {
        self.v.set(~[]);
    }
    pure fn contains_key(key: uint) -> bool {
        contains_key(self, key)
    }
    pure fn contains_key_ref(key: &uint) -> bool {
        contains_key(self, *key)
    }
    pure fn get(key: uint) -> V { get(self, key) }
    pure fn find(key: uint) -> Option<V> { find(self, key) }
    fn rehash() { fail }

    fn update_with_key(key: uint, val: V, ff: fn(uint, V, V) -> V) -> bool {
        match self.find(key) {
            None            => return self.insert(key, val),
            Some(copy orig) => return self.insert(key, ff(key, orig, val)),
        }
    }

    fn update(key: uint, newval: V, ff: fn(V, V) -> V) -> bool {
        return self.update_with_key(key, newval, |_k, v, v1| ff(v,v1));
    }

    pure fn each(it: fn(key: uint, value: V) -> bool) {
        self.each_ref(|k, v| it(*k, *v))
    }
    pure fn each_key(it: fn(key: uint) -> bool) {
        self.each_ref(|k, _v| it(*k))
    }
    pure fn each_value(it: fn(value: V) -> bool) {
        self.each_ref(|_k, v| it(*v))
    }
    pure fn each_ref(it: fn(key: &uint, value: &V) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            match self.v.get_elt(idx) {
              Some(ref elt) => if !it(&idx, elt) { break },
              None => ()
            }
            idx += 1u;
        }
    }
    pure fn each_key_ref(blk: fn(key: &uint) -> bool) {
        self.each_ref(|k, _v| blk(k))
    }
    pure fn each_value_ref(blk: fn(value: &V) -> bool) {
        self.each_ref(|_k, v| blk(v))
    }
}

impl<V: Copy> SmallIntMap<V>: ops::Index<uint, V> {
    pure fn index(key: uint) -> V {
        unsafe {
            get(self, key)
        }
    }
}

/// Cast the given smallintmap to a map::map
pub fn as_map<V: Copy>(s: SmallIntMap<V>) -> map::Map<uint, V> {
    s as map::Map::<uint, V>
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_insert_with_key() {
        let map: SmallIntMap<uint> = mk();

        // given a new key, initialize it with this new count, given
        // given an existing key, add more to its count
        fn addMoreToCount(_k: uint, v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        fn addMoreToCount_simple(v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        // count integers
        map.update(3, 1, addMoreToCount_simple);
        map.update_with_key(9, 1, addMoreToCount);
        map.update(3, 7, addMoreToCount_simple);
        map.update_with_key(5, 3, addMoreToCount);
        map.update_with_key(3, 2, addMoreToCount);

        // check the total counts
        assert 10 == option::get(map.find(3));
        assert  3 == option::get(map.find(5));
        assert  1 == option::get(map.find(9));

        // sadly, no sevens were counted
        assert None == map.find(7);
    }
}
