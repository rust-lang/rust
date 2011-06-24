
export t;
export create;
export union;
export intersect;
export copy;
export clone;
export get;
export equal;
export clear;
export set_all;
export invert;
export difference;
export set;
export is_true;
export is_false;
export to_vec;
export to_str;
export eq_vec;


// FIXME: With recursive object types, we could implement binary methods like
//        union, intersection, and difference. At that point, we could write
//        an optimizing version of this module that produces a different obj
//        for the case where nbits <= 32.

// FIXME: Almost all the functions in this module should be state fns, but the
//        effect system isn't currently working correctly.
type t = rec(vec[mutable uint] storage, uint nbits);


// FIXME: this should be a constant once they work
fn uint_bits() -> uint { ret 32u + (1u << 32u >> 27u); }

fn create(uint nbits, bool init) -> t {
    auto elt = if (init) { !0u } else { 0u };
    auto storage = vec::init_elt_mut[uint](elt, nbits / uint_bits() + 1u);
    ret rec(storage=storage, nbits=nbits);
}

fn process(&fn(uint, uint) -> uint  op, &t v0, &t v1) -> bool {
    auto len = vec::len(v1.storage);
    assert (vec::len(v0.storage) == len);
    assert (v0.nbits == v1.nbits);
    auto changed = false;
    for each (uint i in uint::range(0u, len)) {
        auto w0 = v0.storage.(i);
        auto w1 = v1.storage.(i);
        auto w = op(w0, w1);
        if (w0 != w) { changed = true; v0.storage.(i) = w; }
    }
    ret changed;
}

fn lor(uint w0, uint w1) -> uint { ret w0 | w1; }

fn union(&t v0, &t v1) -> bool { auto sub = lor; ret process(sub, v0, v1); }

fn land(uint w0, uint w1) -> uint { ret w0 & w1; }

fn intersect(&t v0, &t v1) -> bool {
    auto sub = land;
    ret process(sub, v0, v1);
}

fn right(uint w0, uint w1) -> uint { ret w1; }

fn copy(&t v0, t v1) -> bool { auto sub = right; ret process(sub, v0, v1); }

fn clone(t v) -> t {
    auto storage = vec::init_elt_mut[uint](0u, v.nbits / uint_bits() + 1u);
    auto len = vec::len(v.storage);
    for each (uint i in uint::range(0u, len)) { storage.(i) = v.storage.(i); }
    ret rec(storage=storage, nbits=v.nbits);
}

fn get(&t v, uint i) -> bool {
    assert (i < v.nbits);
    auto bits = uint_bits();
    auto w = i / bits;
    auto b = i % bits;
    auto x = 1u & v.storage.(w) >> b;
    ret x == 1u;
}

fn equal(&t v0, &t v1) -> bool {
    // FIXME: when we can break or return from inside an iterator loop,
    //        we can eliminate this painful while-loop

    auto len = vec::len(v1.storage);
    auto i = 0u;
    while (i < len) {
        if (v0.storage.(i) != v1.storage.(i)) { ret false; }
        i = i + 1u;
    }
    ret true;
}

fn clear(&t v) {
    for each (uint i in uint::range(0u, vec::len(v.storage))) {
        v.storage.(i) = 0u;
    }
}

fn set_all(&t v) {
    for each (uint i in uint::range(0u, v.nbits)) { set(v, i, true); }
}

fn invert(&t v) {
    for each (uint i in uint::range(0u, vec::len(v.storage))) {
        v.storage.(i) = !v.storage.(i);
    }
}


/* v0 = v0 - v1 */
fn difference(&t v0, &t v1) -> bool {
    invert(v1);
    auto b = intersect(v0, v1);
    invert(v1);
    ret b;
}

fn set(&t v, uint i, bool x) {
    assert (i < v.nbits);
    auto bits = uint_bits();
    auto w = i / bits;
    auto b = i % bits;
    auto w0 = v.storage.(w);
    auto flag = 1u << b;
    v.storage.(w) =
        if (x) { v.storage.(w) | flag } else { v.storage.(w) & !flag };
}


/* true if all bits are 1 */
fn is_true(&t v) -> bool {
    for (uint i in to_vec(v)) { if (i != 1u) { ret false; } }
    ret true;
}


/* true if all bits are non-1 */
fn is_false(&t v) -> bool {
    for (uint i in to_vec(v)) { if (i == 1u) { ret false; } }
    ret true;
}

fn init_to_vec(t v, uint i) -> uint { ret if (get(v, i)) { 1u } else { 0u }; }

fn to_vec(&t v) -> vec[uint] {
    auto sub = bind init_to_vec(v, _);
    ret vec::init_fn[uint](sub, v.nbits);
}

fn to_str(&t v) -> str {
    auto rs = "";
    for (uint i in bitv::to_vec(v)) {
        if (i == 1u) { rs += "1"; } else { rs += "0"; }
    }
    ret rs;
}


// FIXME: can we just use structural equality on to_vec?
fn eq_vec(&t v0, &vec[uint] v1) -> bool {
    assert (v0.nbits == vec::len[uint](v1));
    auto len = v0.nbits;
    auto i = 0u;
    while (i < len) {
        auto w0 = get(v0, i);
        auto w1 = v1.(i);
        if (!w0 && w1 != 0u || w0 && w1 == 0u) { ret false; }
        i = i + 1u;
    }
    ret true;
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
