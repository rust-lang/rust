
export t;
export create;
export union;
export intersect;
export assign;
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
export to_ivec;
export to_str;
export eq_ivec;


// FIXME: With recursive object types, we could implement binary methods like
//        union, intersection, and difference. At that point, we could write
//        an optimizing version of this module that produces a different obj
//        for the case where nbits <= 32.

type t = @{storage: [mutable uint], nbits: uint};


// FIXME: this should be a constant once they work
fn uint_bits() -> uint { ret 32u + (1u << 32u >> 27u); }

fn create(nbits: uint, init: bool) -> t {
    let elt = if init { !0u } else { 0u };
    let storage = ivec::init_elt_mut[uint](elt, nbits / uint_bits() + 1u);
    ret @{storage: storage, nbits: nbits};
}

fn process(op: &block(uint, uint) -> uint , v0: &t, v1: &t) -> bool {
    let len = ivec::len(v1.storage);
    assert (ivec::len(v0.storage) == len);
    assert (v0.nbits == v1.nbits);
    let changed = false;
    for each i: uint  in uint::range(0u, len) {
        let w0 = v0.storage.(i);
        let w1 = v1.storage.(i);
        let w = op(w0, w1);
        if w0 != w { changed = true; v0.storage.(i) = w; }
    }
    ret changed;
}

fn lor(w0: uint, w1: uint) -> uint { ret w0 | w1; }

fn union(v0: &t, v1: &t) -> bool { let sub = lor; ret process(sub, v0, v1); }

fn land(w0: uint, w1: uint) -> uint { ret w0 & w1; }

fn intersect(v0: &t, v1: &t) -> bool {
    let sub = land;
    ret process(sub, v0, v1);
}

fn right(w0: uint, w1: uint) -> uint { ret w1; }

fn assign(v0: &t, v1: t) -> bool {
    let sub = right;
    ret process(sub, v0, v1);
}

fn clone(v: t) -> t {
    let storage = ivec::init_elt_mut[uint](0u, v.nbits / uint_bits() + 1u);
    let len = ivec::len(v.storage);
    for each i: uint  in uint::range(0u, len) { storage.(i) = v.storage.(i); }
    ret @{storage: storage, nbits: v.nbits};
}

fn get(v: &t, i: uint) -> bool {
    assert (i < v.nbits);
    let bits = uint_bits();
    let w = i / bits;
    let b = i % bits;
    let x = 1u & v.storage.(w) >> b;
    ret x == 1u;
}

fn equal(v0: &t, v1: &t) -> bool {
    // FIXME: when we can break or return from inside an iterator loop,
    //        we can eliminate this painful while-loop

    let len = ivec::len(v1.storage);
    let i = 0u;
    while i < len {
        if v0.storage.(i) != v1.storage.(i) { ret false; }
        i = i + 1u;
    }
    ret true;
}

fn clear(v: &t) {
    for each i: uint  in uint::range(0u, ivec::len(v.storage)) {
        v.storage.(i) = 0u;
    }
}

fn set_all(v: &t) {
    for each i: uint  in uint::range(0u, v.nbits) { set(v, i, true); }
}

fn invert(v: &t) {
    for each i: uint  in uint::range(0u, ivec::len(v.storage)) {
        v.storage.(i) = !v.storage.(i);
    }
}


/* v0 = v0 - v1 */
fn difference(v0: &t, v1: &t) -> bool {
    invert(v1);
    let b = intersect(v0, v1);
    invert(v1);
    ret b;
}

fn set(v: &t, i: uint, x: bool) {
    assert (i < v.nbits);
    let bits = uint_bits();
    let w = i / bits;
    let b = i % bits;
    let flag = 1u << b;
    v.storage.(w) =
        if x { v.storage.(w) | flag } else { v.storage.(w) & !flag };
}


/* true if all bits are 1 */
fn is_true(v: &t) -> bool {
    for i: uint  in to_ivec(v) { if i != 1u { ret false; } }
    ret true;
}


/* true if all bits are non-1 */
fn is_false(v: &t) -> bool {
    for i: uint  in to_ivec(v) { if i == 1u { ret false; } }
    ret true;
}

fn init_to_vec(v: t, i: uint) -> uint { ret if get(v, i) { 1u } else { 0u }; }

fn to_ivec(v: &t) -> [uint] {
    let sub = bind init_to_vec(v, _);
    ret ivec::init_fn[uint](sub, v.nbits);
}

fn to_str(v: &t) -> str {
    let rs = "";
    for i: uint  in to_ivec(v) {
        if i == 1u { rs += "1"; } else { rs += "0"; }
    }
    ret rs;
}

fn eq_ivec(v0: &t, v1: &[uint]) -> bool {
    assert (v0.nbits == ivec::len[uint](v1));
    let len = v0.nbits;
    let i = 0u;
    while i < len {
        let w0 = get(v0, i);
        let w1 = v1.(i);
        if !w0 && w1 != 0u || w0 && w1 == 0u { ret false; }
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
