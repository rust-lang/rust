// An "interner" is a data structure that associates values with uint tags and
// allows bidirectional lookup; i.e. given a value, one can easily find the
// type, and vice versa.
import std::map;
import std::map::{hashmap, hashfn, eqfn};
import dvec::{dvec, extensions};

type interner<T: const> =
    {map: hashmap<T, uint>,
     vect: dvec<T>,
     hasher: hashfn<T>,
     eqer: eqfn<T>};

fn mk<T: const copy>(hasher: hashfn<T>, eqer: eqfn<T>) -> interner<T> {
    let m = map::hashmap::<T, uint>(hasher, eqer);
    ret {map: m, vect: dvec(), hasher: hasher, eqer: eqer};
}

fn intern<T: const copy>(itr: interner<T>, val: T) -> uint {
    alt itr.map.find(val) {
      some(idx) { ret idx; }
      none {
        let new_idx = itr.vect.len();
        itr.map.insert(val, new_idx);
        itr.vect.push(val);
        ret new_idx;
      }
    }
}

// |get| isn't "pure" in the traditional sense, because it can go from
// failing to returning a value as items are interned. But for typestate,
// where we first check a pred and then rely on it, ceasing to fail is ok.
pure fn get<T: const copy>(itr: interner<T>, idx: uint) -> T {
    unchecked {
        itr.vect.get_elt(idx)
    }
}

fn len<T: const>(itr: interner<T>) -> uint { ret itr.vect.len(); }
