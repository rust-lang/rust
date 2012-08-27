/*!
 * A simple map based on a vector for small integer keys. Space requirements
 * are O(highest integer key).
 */
import core::option;
import core::option::{Some, None};
import dvec::DVec;
import map::map;

// FIXME (#2347): Should not be @; there's a bug somewhere in rustc that
// requires this to be.
type smallintmap_<T: copy> = {v: DVec<Option<T>>};

enum smallintmap<T:copy> {
    smallintmap_(@smallintmap_<T>)
}

/// Create a smallintmap
fn mk<T: copy>() -> smallintmap<T> {
    let v = DVec();
    return smallintmap_(@{v: v});
}

/**
 * Add a value to the map. If the map already contains a value for
 * the specified key then the original value is replaced.
 */
#[inline(always)]
fn insert<T: copy>(self: smallintmap<T>, key: uint, val: T) {
    //io::println(fmt!("%?", key));
    self.v.grow_set_elt(key, None, Some(val));
}

/**
 * Get the value for the specified key. If the key does not exist
 * in the map then returns none
 */
pure fn find<T: copy>(self: smallintmap<T>, key: uint) -> Option<T> {
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
pure fn get<T: copy>(self: smallintmap<T>, key: uint) -> T {
    match find(self, key) {
      None => {
        error!("smallintmap::get(): key not present");
        fail;
      }
      Some(v) => return v
    }
}

/// Returns true if the map contains a value for the specified key
fn contains_key<T: copy>(self: smallintmap<T>, key: uint) -> bool {
    return !option::is_none(find(self, key));
}

/// Implements the map::map interface for smallintmap
impl<V: copy> smallintmap<V>: map::map<uint, V> {
    pure fn size() -> uint {
        let mut sz = 0u;
        for self.v.each |item| {
            match item {
              Some(_) => sz += 1u,
              _ => ()
            }
        }
        sz
    }
    #[inline(always)]
    fn insert(+key: uint, +value: V) -> bool {
        let exists = contains_key(self, key);
        insert(self, key, value);
        return !exists;
    }
    fn remove(+key: uint) -> bool {
        if key >= self.v.len() {
            return false;
        }
        let old = self.v.get_elt(key);
        self.v.set_elt(key, None);
        old.is_some()
    }
    fn clear() {
        self.v.set(~[mut]);
    }
    fn contains_key(+key: uint) -> bool {
        contains_key(self, key)
    }
    fn contains_key_ref(key: &uint) -> bool {
        contains_key(self, *key)
    }
    fn get(+key: uint) -> V { get(self, key) }
    pure fn find(+key: uint) -> Option<V> { find(self, key) }
    fn rehash() { fail }
    pure fn each(it: fn(+key: uint, +value: V) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            match self.v.get_elt(idx) {
              Some(elt) => if !it(idx, elt) { break },
              None => ()
            }
            idx += 1u;
        }
    }
    pure fn each_key(it: fn(+key: uint) -> bool) {
        self.each(|k, _v| it(k))
    }
    pure fn each_value(it: fn(+value: V) -> bool) {
        self.each(|_k, v| it(v))
    }
    pure fn each_ref(it: fn(key: &uint, value: &V) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            match self.v.get_elt(idx) {
              Some(elt) => if !it(&idx, &elt) { break },
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

impl<V: copy> smallintmap<V>: ops::index<uint, V> {
    pure fn index(&&key: uint) -> V {
        unchecked {
            get(self, key)
        }
    }
}

/// Cast the given smallintmap to a map::map
fn as_map<V: copy>(s: smallintmap<V>) -> map::map<uint, V> {
    s as map::map::<uint, V>
}
