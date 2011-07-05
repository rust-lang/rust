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

type interner[T] =
    rec(hashmap[T, uint] map,
        mutable vec[T] vect,
        hashfn[T] hasher,
        eqfn[T] eqer);

fn mk[T](hashfn[T] hasher, eqfn[T] eqer) -> interner[T] {
    auto m = map::mk_hashmap[T, uint](hasher, eqer);
    let vec[T] vect = [];
    ret rec(map=m, mutable vect=vect, hasher=hasher, eqer=eqer);
}
fn intern[T](&interner[T] itr, &T val) -> uint {
    alt (itr.map.find(val)) {
        case (some(?idx)) { ret idx; }
        case (none) {
            auto new_idx = vec::len[T](itr.vect);
            itr.map.insert(val, new_idx);
            itr.vect += [val];
            ret new_idx;
        }
    }
}
fn get[T](&interner[T] itr, uint idx) -> T { ret itr.vect.(idx); }
