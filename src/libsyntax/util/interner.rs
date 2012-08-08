// An "interner" is a data structure that associates values with uint tags and
// allows bidirectional lookup; i.e. given a value, one can easily find the
// type, and vice versa.
import std::map;
import std::map::{hashmap, hashfn, eqfn};
import dvec::dvec;

type hash_interner<T: const> =
    {map: hashmap<T, uint>,
     vect: dvec<T>,
     hasher: hashfn<T>,
     eqer: eqfn<T>};

fn mk<T: const copy>(+hasher: hashfn<T>, +eqer: eqfn<T>) -> interner<T> {
    let m = map::hashmap::<T, uint>(copy hasher, copy eqer);
    let hi: hash_interner<T> =
        {map: m, vect: dvec(), hasher: hasher, eqer: eqer};
    return hi as interner::<T>;
}

/* when traits can extend traits, we should extend index<uint,T> to get [] */
trait interner<T: const copy> {
    fn intern(T) -> uint;
    pure fn get(uint) -> T;
    fn len() -> uint;
}

impl <T: const copy> hash_interner<T>: interner<T> {
    fn intern(val: T) -> uint {
        match self.map.find(val) {
          some(idx) => return idx,
          none => {
            let new_idx = self.vect.len();
            self.map.insert(val, new_idx);
            self.vect.push(val);
            return new_idx;
          }
        }
    }

    // this isn't "pure" in the traditional sense, because it can go from
    // failing to returning a value as items are interned. But for typestate,
    // where we first check a pred and then rely on it, ceasing to fail is ok.
    pure fn get(idx: uint) -> T { self.vect.get_elt(idx) }

    fn len() -> uint { return self.vect.len(); }
}