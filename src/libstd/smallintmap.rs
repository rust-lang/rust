/*!
 * A simple map based on a vector for small integer keys. Space requirements
 * are O(highest integer key).
 */
import core::option;
import core::option::{some, none};
import dvec::{dvec, extensions};

// FIXME (#2347): Should not be @; there's a bug somewhere in rustc that
// requires this to be.
type smallintmap<T: copy> = @{v: dvec<option<T>>};

/// Create a smallintmap
fn mk<T: copy>() -> smallintmap<T> {
    ret @{v: dvec()};
}

/**
 * Add a value to the map. If the map already contains a value for
 * the specified key then the original value is replaced.
 */
#[inline(always)]
fn insert<T: copy>(self: smallintmap<T>, key: uint, val: T) {
    self.v.grow_set_elt(key, none, some(val));
}

/**
 * Get the value for the specified key. If the key does not exist
 * in the map then returns none
 */
fn find<T: copy>(self: smallintmap<T>, key: uint) -> option<T> {
    if key < self.v.len() { ret self.v.get_elt(key); }
    ret none::<T>;
}

/**
 * Get the value for the specified key
 *
 * # Failure
 *
 * If the key does not exist in the map
 */
fn get<T: copy>(self: smallintmap<T>, key: uint) -> T {
    alt find(self, key) {
      none { #error("smallintmap::get(): key not present"); fail; }
      some(v) { ret v; }
    }
}

/// Returns true if the map contains a value for the specified key
fn contains_key<T: copy>(self: smallintmap<T>, key: uint) -> bool {
    ret !option::is_none(find(self, key));
}

/// Implements the map::map interface for smallintmap
impl <V: copy> of map::map<uint, V> for smallintmap<V> {
    fn size() -> uint {
        let mut sz = 0u;
        for self.v.each |item| {
            alt item { some(_) { sz += 1u; } _ {} }
        }
        sz
    }
    #[inline(always)]
    fn insert(+key: uint, +value: V) -> bool {
        let exists = contains_key(self, key);
        insert(self, key, value);
        ret !exists;
    }
    fn remove(&&key: uint) -> option<V> {
        if key >= self.v.len() { ret none; }
        let old = self.v.get_elt(key);
        self.v.set_elt(key, none);
        old
    }
    fn clear() {
        self.v.set(~[mut]);
    }
    fn contains_key(&&key: uint) -> bool {
        contains_key(self, key)
    }
    fn get(&&key: uint) -> V { get(self, key) }
    fn [](&&key: uint) -> V { get(self, key) }
    fn find(&&key: uint) -> option<V> { find(self, key) }
    fn rehash() { fail }
    fn each(it: fn(&&uint, V) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            alt self.v.get_elt(idx) {
              some(elt) {
                if !it(idx, elt) { break; }
              }
              none { }
            }
            idx += 1u;
        }
    }
    fn each_key(it: fn(&&uint) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            if self.v.get_elt(idx) != none && !it(idx) { ret; }
            idx += 1u;
        }
    }
    fn each_value(it: fn(V) -> bool) {
        self.each(|_i, v| it(v));
    }
}

/// Cast the given smallintmap to a map::map
fn as_map<V: copy>(s: smallintmap<V>) -> map::map<uint, V> {
    s as map::map::<uint, V>
}
