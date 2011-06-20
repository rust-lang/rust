

/// A simple map based on a vector for small integer keys. Space requirements
/// are O(highest integer key).
import option::none;
import option::some;

// FIXME: Should not be @; there's a bug somewhere in rustc that requires this
// to be.
type smallintmap[T] = @rec(mutable (option::t[T])[mutable] v);

fn mk[T]() -> smallintmap[T] {
    let (option::t[T])[mutable] v = ~[mutable];
    ret @rec(mutable v=v);
}

fn insert[T](&smallintmap[T] m, uint key, &T val) {
    ivec::grow_set[option::t[T]](m.v, key, none[T], some[T](val));
}

fn find[T](&smallintmap[T] m, uint key) -> option::t[T] {
    if (key < ivec::len[option::t[T]](m.v)) { ret m.v.(key); }
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
    m.v = ivec::slice_mut[option::t[T]](m.v, 0u, len);
}

fn max_key[T](&smallintmap[T] m) -> uint { ret ivec::len[option::t[T]](m.v); }

