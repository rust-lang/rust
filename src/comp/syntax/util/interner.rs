// An "interner" is a data structure that associates values with uint tags and
// allows bidirectional lookup; i.e. given a value, one can easily find the
// type, and vice versa.
import std::vec;
import std::map;
import std::map::hashmap;
import std::map::hashfn;
import std::map::eqfn;
import std::option;
import std::option::none;
import std::option::some;

type interner<T> =
    {map: hashmap<T, uint>,
     mutable vect: [T],
     hasher: hashfn<T>,
     eqer: eqfn<T>};

fn mk<@T>(hasher: hashfn<T>, eqer: eqfn<T>) -> interner<T> {
    let m = map::mk_hashmap[T, uint](hasher, eqer);
    ret {map: m, mutable vect: ~[], hasher: hasher, eqer: eqer};
}

fn intern<@T>(itr: &interner<T>, val: &T) -> uint {
    alt itr.map.find(val) {
      some(idx) { ret idx; }
      none. {
        let new_idx = vec::len[T](itr.vect);
        itr.map.insert(val, new_idx);
        itr.vect += ~[val];
        ret new_idx;
      }
    }
}

fn get<T>(itr: &interner<T>, idx: uint) -> T { ret itr.vect.(idx); }

fn len<T>(itr : &interner<T>) -> uint { ret vec::len(itr.vect); }

