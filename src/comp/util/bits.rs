import std._uint;
import std._int;
import std._vec;

// FIXME: With recursive object types, we could implement binary methods like
//        union, intersection, and difference. At that point, we could write
//        an optimizing version of this module that produces a different obj
//        for the case where nbits < 32.

state type t = rec(vec[mutable uint] storage, uint nbits);

// FIXME: this should be a constant once they work
fn uint_bits() -> uint {
    ret 32u + ((1u << 32u) >> 27u) - 1u;
}

// FIXME: this should be state
fn create(uint nbits, bool init) -> t {
    auto elt;
    if (init) {
        elt = 1u;
    } else {
        elt = 0u;
    }

    ret rec(storage = _vec.init_elt[mutable uint](nbits / uint_bits() + 1u, elt),
            nbits = nbits);
}

// FIXME: this should be state
fn process(fn(uint, uint) -> uint op, t v0, t v1) -> bool {
    auto len = _vec.len[mutable uint](v1.storage);

    check (_vec.len[mutable uint](v0.storage) == len);
    check (v0.nbits == v1.nbits);

    auto changed = false;

    for each (uint i in _uint.range(0u, len)) {
        auto w0 = v0.storage.(i);
        auto w1 = v1.storage.(i);

        auto w = op(w0, w1);
        if (w0 != w) {
            changed = true;
            v0.storage.(i) = w;
        }
    }

    ret changed;
}

fn lor(uint w0, uint w1) -> uint {
    ret w0 | w1;
}

// FIXME: this should be state
fn union(t v0, t v1) -> bool {
    auto sub = lor;
    ret process(sub, v0, v1);
}

fn land(uint w0, uint w1) -> uint {
    ret w0 & w1;
}

// FIXME: this should be state
fn intersect(t v0, t v1) -> bool {
    auto sub = land;
    ret process(sub, v0, v1);
}

fn right(uint w0, uint w1) -> uint {
    ret w1;
}

// FIXME: this should be state
fn copy(t v0, t v1) -> bool {
    auto sub = right;
    ret process(sub, v0, v1);
}

// FIXME: this should be state
fn get(t v, uint i) -> bool {
    check (i < v.nbits);

    auto bits = uint_bits();

    auto w = i / bits;
    auto b = i % bits;
    auto x = 1u & (v.storage.(w) >> b);
    ret x == 1u;
}

// FIXME: this should be state
fn equal(t v0, t v1) -> bool {
    // FIXME: when we can break or return from inside an iterator loop,
    //        we can eliminate this painful while-loop
    auto len = _vec.len[mutable uint](v1.storage);
    auto i = 0u;
    while (i < len) {
        if (v0.storage.(i) != v1.storage.(i)) {
            ret false;
        }
        i = i + 1u;
    }
    ret true;
}

// FIXME: this should be state
fn clear(t v) {
    for each (uint i in _uint.range(0u, _vec.len[mutable uint](v.storage))) {
        v.storage.(i) = 0u;
    }
}

// FIXME: this should be state
fn invert(t v) {
    for each (uint i in _uint.range(0u, _vec.len[mutable uint](v.storage))) {
        v.storage.(i) = ~v.storage.(i);
    }
}

// FIXME: this should be state
/* v0 = v0 - v1 */
fn difference(t v0, t v1) -> bool {
    invert(v1);
    auto b = intersect(v0, v1);
    invert(v1);
    ret b;
}

// FIXME: this should be state
fn set(t v, uint i, bool x) {
    check (i < v.nbits);

    auto bits = uint_bits();

    auto w = i / bits;
    auto b = i % bits;
    auto w0 = v.storage.(w);
    auto flag = 1u << b;
    if (x) {
        v.storage.(w) = v.storage.(w) | flag;
    } else {
        v.storage.(w) = v.storage.(w) & ~flag;
    }
}

// FIXME: this should be state
fn init_to_vec(t v, uint i) -> uint {
    if (get(v, i)) {
        ret 1u;
    } else {
        ret 0u;
    }
}

// FIXME: this should be state
fn to_vec(t v) -> vec[uint] {
    auto sub = bind init_to_vec(v, _);
    ret _vec.init_fn[uint](sub, v.nbits);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
