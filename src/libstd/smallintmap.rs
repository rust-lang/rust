#[doc = "
A simple map based on a vector for small integer keys. Space requirements
are O(highest integer key).
"];
import core::option;
import core::option::{some, none};

// FIXME: Should not be @; there's a bug somewhere in rustc that requires this
// to be.
type smallintmap<T: copy> = @{mut v: [mut option<T>]};

#[doc = "Create a smallintmap"]
fn mk<T: copy>() -> smallintmap<T> {
    let v: [mut option<T>] = [mut];
    ret @{mut v: v};
}

#[doc = "
Add a value to the map. If the map already contains a value for
the specified key then the original value is replaced.
"]
fn insert<T: copy>(m: smallintmap<T>, key: uint, val: T) {
    vec::grow_set::<option<T>>(m.v, key, none::<T>, some::<T>(val));
}

#[doc = "
Get the value for the specified key. If the key does not exist
in the map then returns none
"]
fn find<T: copy>(m: smallintmap<T>, key: uint) -> option<T> {
    if key < vec::len::<option<T>>(m.v) { ret m.v[key]; }
    ret none::<T>;
}

#[doc = "
Get the value for the specified key

# Failure

If the key does not exist in the map
"]
fn get<T: copy>(m: smallintmap<T>, key: uint) -> T {
    alt find(m, key) {
      none { #error("smallintmap::get(): key not present"); fail; }
      some(v) { ret v; }
    }
}

#[doc = "
Returns true if the map contains a value for the specified key
"]
fn contains_key<T: copy>(m: smallintmap<T>, key: uint) -> bool {
    ret !option::is_none(find::<T>(m, key));
}

// FIXME: Are these really useful?

fn truncate<T: copy>(m: smallintmap<T>, len: uint) {
    m.v = vec::to_mut(vec::slice::<option<T>>(m.v, 0u, len));
}

fn max_key<T: copy>(m: smallintmap<T>) -> uint {
    ret vec::len::<option<T>>(m.v);
}

#[doc = "Implements the map::map interface for smallintmap"]
impl <V: copy> of map::map<uint, V> for smallintmap<V> {
    fn size() -> uint {
        let mut sz = 0u;
        for vec::each(self.v) {|item|
            alt item { some(_) { sz += 1u; } _ {} }
        }
        sz
    }
    fn insert(&&key: uint, value: V) -> bool {
        let exists = contains_key(self, key);
        insert(self, key, value);
        ret !exists;
    }
    fn remove(&&key: uint) -> option<V> {
        if key >= vec::len(self.v) { ret none; }
        let old = self.v[key];
        self.v[key] = none;
        old
    }
    fn contains_key(&&key: uint) -> bool {
        contains_key(self, key)
    }
    fn get(&&key: uint) -> V { get(self, key) }
    fn find(&&key: uint) -> option<V> { find(self, key) }
    fn rehash() { fail }
    fn each(it: fn(&&uint, V) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            alt self.v[idx] {
              some(elt) {
                if !it(idx, copy elt) { break; }
              }
              none { }
            }
            idx += 1u;
        }
    }
    fn each_key(it: fn(&&uint) -> bool) {
        let mut idx = 0u, l = self.v.len();
        while idx < l {
            if self.v[idx] != none && !it(idx) { ret; }
            idx += 1u;
        }
    }
    fn each_value(it: fn(V) -> bool) {
        self.each {|_i, v| it(v)}
    }
}

#[doc = "Cast the given smallintmap to a map::map"]
fn as_map<V: copy>(s: smallintmap<V>) -> map::map<uint, V> {
    s as map::map::<uint, V>
}
