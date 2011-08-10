

/// A simple map based on a vector for small integer keys. Space requirements
/// are O(highest integer key).
import option::none;
import option::some;

// FIXME: Should not be @; there's a bug somewhere in rustc that requires this
// to be.
type smallintmap[T] = @{mutable v: [mutable option::t<T>]};

fn mk[@T]() -> smallintmap<T> {
    let v: [mutable option::t<T>] = ~[mutable];
    ret @{mutable v: v};
}

fn insert[@T](m: &smallintmap<T>, key: uint, val: &T) {
    vec::grow_set[option::t<T>](m.v, key, none[T], some[T](val));
}

fn find[@T](m: &smallintmap<T>, key: uint) -> option::t<T> {
    if key < vec::len[option::t<T>](m.v) { ret m.v.(key); }
    ret none[T];
}

fn get[@T](m: &smallintmap<T>, key: uint) -> T {
    alt find[T](m, key) {
      none[T]. { log_err "smallintmap::get(): key not present"; fail; }
      some[T](v) { ret v; }
    }
}

fn contains_key[@T](m: &smallintmap<T>, key: uint) -> bool {
    ret !option::is_none(find[T](m, key));
}

fn truncate[@T](m: &smallintmap<T>, len: uint) {
    m.v = vec::slice_mut[option::t<T>](m.v, 0u, len);
}

fn max_key[T](m: &smallintmap<T>) -> uint {
    ret vec::len[option::t<T>](m.v);
}

