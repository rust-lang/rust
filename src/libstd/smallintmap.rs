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
type smallintmap<T> = @{mutable v: [mutable option::t<T>]};

/*
Function: mk

Create a smallintmap
*/
fn mk<T>() -> smallintmap<T> {
    let v: [mutable option::t<T>] = [mutable];
    ret @{mutable v: v};
}

/*
Function: insert

Add a value to the map. If the map already contains a value for
the specified key then the original value is replaced.
*/
fn insert<T: copy>(m: smallintmap<T>, key: uint, val: T) {
    vec::grow_set::<option::t<T>>(m.v, key, none::<T>, some::<T>(val));
}

/*
Function: find

Get the value for the specified key. If the key does not exist
in the map then returns none.
*/
fn find<T: copy>(m: smallintmap<T>, key: uint) -> option::t<T> {
    if key < vec::len::<option::t<T>>(m.v) { ret m.v[key]; }
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
      none. { #error("smallintmap::get(): key not present"); fail; }
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
    m.v = vec::slice_mut::<option::t<T>>(m.v, 0u, len);
}

fn max_key<T>(m: smallintmap<T>) -> uint {
    ret vec::len::<option::t<T>>(m.v);
}

