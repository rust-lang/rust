/// A simple map based on a vector for small integer keys. Space requirements
/// are O(highest integer key).

import option::none;
import option::some;

type smallintmap[T] = rec(mutable vec[mutable option::t[T]] v);

fn mk[T]() -> smallintmap[T] {
    let vec[mutable option::t[T]] v = [mutable];
    ret rec(mutable v=v);
}

fn insert[T](&smallintmap[T] m, uint key, &T val) {
    vec::grow_set[option::t[T]](m.v, key, none[T], some[T](val));
}

fn find[T](&smallintmap[T] m, uint key) -> option::t[T] {
    if (key < vec::len[option::t[T]](m.v)) { ret m.v.(key); }
    ret none[T];
}

fn get[T](&smallintmap[T] m, uint key) -> T {
    alt (find[T](m, key)) {
        case (none[T]) {
            log_err "smallintmap::get(): key not present";
            fail;
        }
        case (some[T](?v)) { ret v; }
    }
}

fn contains_key[T](&smallintmap[T] m, uint key) -> bool {
    ret !option::is_none(find[T](m, key));
}

fn truncate[T](&smallintmap[T] m, uint len) {
    m.v = vec::slice_mut[option::t[T]](m.v, 0u, len);
}

fn max_key[T](&smallintmap[T] m) -> uint {
    ret vec::len[option::t[T]](m.v);
}

