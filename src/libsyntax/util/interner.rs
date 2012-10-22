// An "interner" is a data structure that associates values with uint tags and
// allows bidirectional lookup; i.e. given a value, one can easily find the
// type, and vice versa.
use std::map;
use std::map::HashMap;
use dvec::DVec;
use cmp::Eq;
use hash::Hash;
use to_bytes::IterBytes;

type hash_interner<T: Const> =
    {map: HashMap<T, uint>,
     vect: DVec<T>};

fn mk<T:Eq IterBytes Hash Const Copy>() -> Interner<T> {
    let m = map::HashMap::<T, uint>();
    let hi: hash_interner<T> =
        {map: m, vect: DVec()};
    move ((move hi) as Interner::<T>)
}

fn mk_prefill<T:Eq IterBytes Hash Const Copy>(init: ~[T]) -> Interner<T> {
    let rv = mk();
    for init.each() |v| { rv.intern(*v); }
    return rv;
}


/* when traits can extend traits, we should extend index<uint,T> to get [] */
trait Interner<T:Eq IterBytes Hash Const Copy> {
    fn intern(T) -> uint;
    fn gensym(T) -> uint;
    pure fn get(uint) -> T;
    fn len() -> uint;
}

impl <T:Eq IterBytes Hash Const Copy> hash_interner<T>: Interner<T> {
    fn intern(val: T) -> uint {
        match self.map.find(val) {
          Some(idx) => return idx,
          None => {
            let new_idx = self.vect.len();
            self.map.insert(val, new_idx);
            self.vect.push(val);
            return new_idx;
          }
        }
    }
    fn gensym(val: T) -> uint {
        let new_idx = self.vect.len();
        // leave out of .map to avoid colliding
        self.vect.push(val);
        return new_idx;
    }

    // this isn't "pure" in the traditional sense, because it can go from
    // failing to returning a value as items are interned. But for typestate,
    // where we first check a pred and then rely on it, ceasing to fail is ok.
    pure fn get(idx: uint) -> T { self.vect.get_elt(idx) }

    fn len() -> uint { return self.vect.len(); }
}
