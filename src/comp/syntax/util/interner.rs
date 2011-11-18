// An "interner" is a data structure that associates values with uint tags and
// allows bidirectional lookup; i.e. given a value, one can easily find the
// type, and vice versa.
import std::{vec, map};
import std::map::{hashmap, hashfn, eqfn};
import std::option::{none, some};

type interner<T> =
    {map: hashmap<T, uint>,
     mutable vect: [T],
     hasher: hashfn<T>,
     eqer: eqfn<T>};

fn mk<copy T>(hasher: hashfn<T>, eqer: eqfn<T>) -> interner<T> {
    let m = map::mk_hashmap::<T, uint>(hasher, eqer);
    ret {map: m, mutable vect: [], hasher: hasher, eqer: eqer};
}

fn intern<copy T>(itr: interner<T>, val: T) -> uint {
    alt itr.map.find(val) {
      some(idx) { ret idx; }
      none. {
        let new_idx = vec::len::<T>(itr.vect);
        itr.map.insert(val, new_idx);
        itr.vect += [val];
        ret new_idx;
      }
    }
}

// |get| isn't "pure" in the traditional sense, because it can go from
// failing to returning a value as items are interned. But for typestate,
// where we first check a pred and then rely on it, ceasing to fail is ok.
pure fn get<copy T>(itr: interner<T>, idx: uint) -> T {
    unchecked {
        itr.vect[idx]
    }
}

fn len<T>(itr: interner<T>) -> uint { ret vec::len(itr.vect); }
