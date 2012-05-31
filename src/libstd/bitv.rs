export bitv;
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
export to_vec;
export to_str;
export eq_vec;

// FIXME: With recursive object types, we could implement binary methods like
//        union, intersection, and difference. At that point, we could write
//        an optimizing version of this module that produces a different obj
//        for the case where nbits <= 32.
// (Issue #2341)

#[doc = "The bitvector type"]
type bitv = @{storage: [mut uint], nbits: uint};

const uint_bits: uint = 32u + (1u << 32u >> 27u);

#[doc = "
Constructs a bitvector

# Arguments

* nbits - The number of bits in the bitvector
* init - If true then the bits are initialized to 1, otherwise 0
"]
fn bitv(nbits: uint, init: bool) -> bitv {
    let elt = if init { !0u } else { 0u };
    let storage = vec::to_mut(vec::from_elem(nbits / uint_bits + 1u, elt));
    ret @{storage: storage, nbits: nbits};
}

fn process(v0: bitv, v1: bitv, op: fn(uint, uint) -> uint) -> bool {
    let len = vec::len(v1.storage);
    assert (vec::len(v0.storage) == len);
    assert (v0.nbits == v1.nbits);
    let mut changed = false;
    for uint::range(0u, len) {|i|
        let w0 = v0.storage[i];
        let w1 = v1.storage[i];
        let w = op(w0, w1);
        if w0 != w { changed = true; v0.storage[i] = w; }
    };
    ret changed;
}


fn lor(w0: uint, w1: uint) -> uint { ret w0 | w1; }

fn union(v0: bitv, v1: bitv) -> bool {
    let sub = lor; ret process(v0, v1, sub);
}

fn land(w0: uint, w1: uint) -> uint { ret w0 & w1; }

#[doc = "
Calculates the intersection of two bitvectors

Sets `v0` to the intersection of `v0` and `v1`. Both bitvectors must be the
same length. Returns 'true' if `v0` was changed.
"]
fn intersect(v0: bitv, v1: bitv) -> bool {
    let sub = land;
    ret process(v0, v1, sub);
}

fn right(_w0: uint, w1: uint) -> uint { ret w1; }

#[doc = "
Assigns the value of `v1` to `v0`

Both bitvectors must be the same length. Returns `true` if `v0` was changed
"]
fn assign(v0: bitv, v1: bitv) -> bool {
    let sub = right; ret process(v0, v1, sub);
}

#[doc = "Makes a copy of a bitvector"]
fn clone(v: bitv) -> bitv {
    let storage = vec::to_mut(vec::from_elem(v.nbits / uint_bits + 1u, 0u));
    let len = vec::len(v.storage);
    for uint::range(0u, len) {|i| storage[i] = v.storage[i]; };
    ret @{storage: storage, nbits: v.nbits};
}

#[doc = "Retreive the value at index `i`"]
#[inline(always)]
pure fn get(v: bitv, i: uint) -> bool {
    assert (i < v.nbits);
    let bits = uint_bits;
    let w = i / bits;
    let b = i % bits;
    let x = 1u & v.storage[w] >> b;
    ret x == 1u;
}

#[doc = "
Compares two bitvectors

Both bitvectors must be the same length. Returns `true` if both bitvectors
contain identical elements.
"]
fn equal(v0: bitv, v1: bitv) -> bool {
    if v0.nbits != v1.nbits { ret false; }
    let len = vec::len(v1.storage);
    for uint::iterate(0u, len) {|i|
        if v0.storage[i] != v1.storage[i] { ret false; }
    }
}

#[doc = "Set all bits to 0"]
#[inline(always)]
fn clear(v: bitv) { for each_storage(v) {|w| w = 0u } }

#[doc = "Set all bits to 1"]
#[inline(always)]
fn set_all(v: bitv) { for each_storage(v) {|w| w = !0u } }

#[doc = "Invert all bits"]
#[inline(always)]
fn invert(v: bitv) { for each_storage(v) {|w| w = !w } }

#[doc = "
Calculate the difference between two bitvectors

Sets each element of `v0` to the value of that element minus the element
of `v1` at the same index. Both bitvectors must be the same length.

Returns `true` if `v0` was changed.
"]
fn difference(v0: bitv, v1: bitv) -> bool {
    invert(v1);
    let b = intersect(v0, v1);
    invert(v1);
    ret b;
}

#[doc = "
Set the value of a bit at a given index

`i` must be less than the length of the bitvector.
"]
#[inline(always)]
fn set(v: bitv, i: uint, x: bool) {
    assert (i < v.nbits);
    let bits = uint_bits;
    let w = i / bits;
    let b = i % bits;
    let flag = 1u << b;
    v.storage[w] = if x { v.storage[w] | flag } else { v.storage[w] & !flag };
}


#[doc = "Returns true if all bits are 1"]
fn is_true(v: bitv) -> bool {
    for each(v) {|i| if !i { ret false; } }
    ret true;
}


#[doc = "Returns true if all bits are 0"]
fn is_false(v: bitv) -> bool {
    for each(v) {|i| if i { ret false; } }
    ret true;
}

fn init_to_vec(v: bitv, i: uint) -> uint {
    ret if get(v, i) { 1u } else { 0u };
}

#[doc = "
Converts the bitvector to a vector of uint with the same length.

Each uint in the resulting vector has either value 0u or 1u.
"]
fn to_vec(v: bitv) -> [uint] {
    let sub = bind init_to_vec(v, _);
    ret vec::from_fn::<uint>(v.nbits, sub);
}

#[inline(always)]
fn each(v: bitv, f: fn(bool) -> bool) {
    let mut i = 0u;
    while i < v.nbits {
        if !f(get(v, i)) { break; }
        i = i + 1u;
    }
}

#[inline(always)]
fn each_storage(v: bitv, op: fn(&uint) -> bool) {
    for uint::range(0u, vec::len(v.storage)) {|i|
        let mut w = v.storage[i];
        let b = !op(w);
        v.storage[i] = w;
        if !b { break; }
    }
}

#[doc = "
Converts the bitvector to a string.

The resulting string has the same length as the bitvector, and each character
is either '0' or '1'.
"]
fn to_str(v: bitv) -> str {
    let mut rs = "";
    for each(v) {|i| if i { rs += "1"; } else { rs += "0"; } }
    ret rs;
}

#[doc = "
Compare a bitvector to a vector of uint

The uint vector is expected to only contain the values 0u and 1u. Both the
bitvector and vector must have the same length
"]
fn eq_vec(v0: bitv, v1: [uint]) -> bool {
    assert (v0.nbits == vec::len::<uint>(v1));
    let len = v0.nbits;
    let mut i = 0u;
    while i < len {
        let w0 = get(v0, i);
        let w1 = v1[i];
        if !w0 && w1 != 0u || w0 && w1 == 0u { ret false; }
        i = i + 1u;
    }
    ret true;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_to_str() {
        let zerolen = bitv(0u, false);
        assert to_str(zerolen) == "";

        let eightbits = bitv(8u, false);
        assert to_str(eightbits) == "00000000";
    }

    #[test]
    fn test_0_elements() {
        let mut act;
        let mut exp;
        act = bitv(0u, false);
        exp = vec::from_elem::<uint>(0u, 0u);
        assert (eq_vec(act, exp));
    }

    #[test]
    fn test_1_element() {
        let mut act;
        act = bitv(1u, false);
        assert (eq_vec(act, [0u]));
        act = bitv(1u, true);
        assert (eq_vec(act, [1u]));
    }

    #[test]
    fn test_10_elements() {
        let mut act;
        // all 0

        act = bitv(10u, false);
        assert (eq_vec(act, [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(10u, true);
        assert (eq_vec(act, [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(10u, false);
        set(act, 0u, true);
        set(act, 1u, true);
        set(act, 2u, true);
        set(act, 3u, true);
        set(act, 4u, true);
        assert (eq_vec(act, [1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(10u, false);
        set(act, 5u, true);
        set(act, 6u, true);
        set(act, 7u, true);
        set(act, 8u, true);
        set(act, 9u, true);
        assert (eq_vec(act, [0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(10u, false);
        set(act, 0u, true);
        set(act, 3u, true);
        set(act, 6u, true);
        set(act, 9u, true);
        assert (eq_vec(act, [1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 1u]));
    }

    #[test]
    fn test_31_elements() {
        let mut act;
        // all 0

        act = bitv(31u, false);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(31u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(31u, false);
        set(act, 0u, true);
        set(act, 1u, true);
        set(act, 2u, true);
        set(act, 3u, true);
        set(act, 4u, true);
        set(act, 5u, true);
        set(act, 6u, true);
        set(act, 7u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(31u, false);
        set(act, 16u, true);
        set(act, 17u, true);
        set(act, 18u, true);
        set(act, 19u, true);
        set(act, 20u, true);
        set(act, 21u, true);
        set(act, 22u, true);
        set(act, 23u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(31u, false);
        set(act, 24u, true);
        set(act, 25u, true);
        set(act, 26u, true);
        set(act, 27u, true);
        set(act, 28u, true);
        set(act, 29u, true);
        set(act, 30u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(31u, false);
        set(act, 3u, true);
        set(act, 17u, true);
        set(act, 30u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u]));
    }

    #[test]
    fn test_32_elements() {
        let mut act;
        // all 0

        act = bitv(32u, false);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(32u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(32u, false);
        set(act, 0u, true);
        set(act, 1u, true);
        set(act, 2u, true);
        set(act, 3u, true);
        set(act, 4u, true);
        set(act, 5u, true);
        set(act, 6u, true);
        set(act, 7u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(32u, false);
        set(act, 16u, true);
        set(act, 17u, true);
        set(act, 18u, true);
        set(act, 19u, true);
        set(act, 20u, true);
        set(act, 21u, true);
        set(act, 22u, true);
        set(act, 23u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(32u, false);
        set(act, 24u, true);
        set(act, 25u, true);
        set(act, 26u, true);
        set(act, 27u, true);
        set(act, 28u, true);
        set(act, 29u, true);
        set(act, 30u, true);
        set(act, 31u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(32u, false);
        set(act, 3u, true);
        set(act, 17u, true);
        set(act, 30u, true);
        set(act, 31u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 1u]));
    }

    #[test]
    fn test_33_elements() {
        let mut act;
        // all 0

        act = bitv(33u, false);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = bitv(33u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = bitv(33u, false);
        set(act, 0u, true);
        set(act, 1u, true);
        set(act, 2u, true);
        set(act, 3u, true);
        set(act, 4u, true);
        set(act, 5u, true);
        set(act, 6u, true);
        set(act, 7u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(33u, false);
        set(act, 16u, true);
        set(act, 17u, true);
        set(act, 18u, true);
        set(act, 19u, true);
        set(act, 20u, true);
        set(act, 21u, true);
        set(act, 22u, true);
        set(act, 23u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = bitv(33u, false);
        set(act, 24u, true);
        set(act, 25u, true);
        set(act, 26u, true);
        set(act, 27u, true);
        set(act, 28u, true);
        set(act, 29u, true);
        set(act, 30u, true);
        set(act, 31u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 0u]));
        // mixed

        act = bitv(33u, false);
        set(act, 3u, true);
        set(act, 17u, true);
        set(act, 30u, true);
        set(act, 31u, true);
        set(act, 32u, true);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 1u, 1u, 1u]));
    }

    #[test]
    fn test_equal_differing_sizes() {
        let v0 = bitv(10u, false);
        let v1 = bitv(11u, false);
        assert !equal(v0, v1);
    }

    #[test]
    fn test_equal_greatly_differing_sizes() {
        let v0 = bitv(10u, false);
        let v1 = bitv(110u, false);
        assert !equal(v0, v1);
    }

}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
