/*
Module: smallintmap

A simple map based on a vector for small integer keys. Space requirements
are O(highest integer key).
*/
import core::option;
import core::option::{some, none};

// FIXME: Should not be @; there's a bug somewhere in rustc that requires this
// to be.
/*
Type: smallintmap
*/
type smallintmap<T> = @{mutable v: [mutable option<T>]};

/*
Function: mk

Create a smallintmap
*/
fn mk<T>() -> smallintmap<T> {
    let v: [mutable option<T>] = [mutable];
    ret @{mutable v: v};
}

/*
Function: insert

Add a value to the map. If the map already contains a value for
the specified key then the original value is replaced.
*/
fn insert<T: copy>(m: smallintmap<T>, key: uint, val: T) {
    vec::grow_set::<option<T>>(m.v, key, none::<T>, some::<T>(val));
}

/*
Function: find

Get the value for the specified key. If the key does not exist
in the map then returns none
*/
fn find<T: copy>(m: smallintmap<T>, key: uint) -> option<T> {
    if key < vec::len::<option<T>>(m.v) { ret m.v[key]; }
    ret none::<T>;
}

/*
Method: get

Get the value for the specified key

Failure:

If the key does not exist in the map
*/
fn get<T: copy>(m: smallintmap<T>, key: uint) -> T {
    alt find(m, key) {
      none { #error("smallintmap::get(): key not present"); fail; }
      some(v) { ret v; }
    }
}

/*
Method: contains_key

Returns true if the map contains a value for the specified key
*/
fn contains_key<T: copy>(m: smallintmap<T>, key: uint) -> bool {
    ret !option::is_none(find::<T>(m, key));
}

// FIXME: Are these really useful?

fn truncate<T: copy>(m: smallintmap<T>, len: uint) {
    m.v = vec::slice_mut::<option<T>>(m.v, 0u, len);
}

fn max_key<T>(m: smallintmap<T>) -> uint {
    ret vec::len::<option<T>>(m.v);
}

/*
Impl: map

Implements the map::map interface for smallintmap
*/
impl <V: copy> of map::map<uint, V> for smallintmap<V> {
    fn size() -> uint {
        let sz = 0u;
        for item in self.v {
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
    fn items(it: fn(&&uint, V)) {
        let idx = 0u;
        for item in self.v {
            alt item {
              some(elt) {
                it(idx, elt);
              }
              none { }
            }
            idx += 1u;
        }
    }
    fn keys(it: fn(&&uint)) {
        let idx = 0u;
        for item in self.v {
            if item != none { it(idx); }
            idx += 1u;
        }
    }
    fn values(it: fn(V)) {
        for item in self.v {
            alt item { some(elt) { it(elt); } _ {} }
        }
    }
}

/*
Funtion: as_map

Cast the given smallintmap to a map::map
*/
fn as_map<V>(s: smallintmap<V>) -> map::map<uint, V> {
    s as map::map::<uint, V>
}
