/*
Module: bitv

Bitvectors.
*/

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
export to_vec;
export to_str;
export eq_vec;

// FIXME: With recursive object types, we could implement binary methods like
//        union, intersection, and difference. At that point, we could write
//        an optimizing version of this module that produces a different obj
//        for the case where nbits <= 32.

/*
Type: t

The bitvector type.
*/
type t = @{storage: [mutable uint], nbits: uint};


const uint_bits: uint = 32u + (1u << 32u >> 27u);

/*
Function: create

Constructs a bitvector.

Parameters:
nbits - The number of bits in the bitvector
init - If true then the bits are initialized to 1, otherwise 0
*/
fn create(nbits: uint, init: bool) -> t {
    let elt = if init { !0u } else { 0u };
    let storage = vec::init_elt_mut::<uint>(elt, nbits / uint_bits + 1u);
    ret @{storage: storage, nbits: nbits};
}

fn process(v0: t, v1: t, op: block(uint, uint) -> uint) -> bool {
    let len = vec::len(v1.storage);
    assert (vec::len(v0.storage) == len);
    assert (v0.nbits == v1.nbits);
    let changed = false;
    uint::range(0u, len) {|i|
        let w0 = v0.storage[i];
        let w1 = v1.storage[i];
        let w = op(w0, w1);
        if w0 != w { changed = true; v0.storage[i] = w; }
    };
    ret changed;
}


fn lor(w0: uint, w1: uint) -> uint { ret w0 | w1; }

fn union(v0: t, v1: t) -> bool { let sub = lor; ret process(v0, v1, sub); }

fn land(w0: uint, w1: uint) -> uint { ret w0 & w1; }

/*
Function: intersect

Calculates the intersection of two bitvectors

Sets `v0` to the intersection of `v0` and `v1`

Preconditions:

Both bitvectors must be the same length

Returns:

True if `v0` was changed
*/
fn intersect(v0: t, v1: t) -> bool {
    let sub = land;
    ret process(v0, v1, sub);
}

fn right(_w0: uint, w1: uint) -> uint { ret w1; }

/*
Function: assign

Assigns the value of `v1` to `v0`

Preconditions:

Both bitvectors must be the same length

Returns:

True if `v0` was changed
*/
fn assign(v0: t, v1: t) -> bool { let sub = right; ret process(v0, v1, sub); }

/*
Function: clone

Makes a copy of a bitvector
*/
fn clone(v: t) -> t {
    let storage = vec::init_elt_mut::<uint>(0u, v.nbits / uint_bits + 1u);
    let len = vec::len(v.storage);
    uint::range(0u, len) {|i| storage[i] = v.storage[i]; };
    ret @{storage: storage, nbits: v.nbits};
}

/*
Function: get

Retreive the value at index `i`
*/
pure fn get(v: t, i: uint) -> bool {
    assert (i < v.nbits);
    let bits = uint_bits;
    let w = i / bits;
    let b = i % bits;
    let x = 1u & v.storage[w] >> b;
    ret x == 1u;
}

// FIXME: This doesn't account for the actual size of the vectors,
// so it could end up comparing garbage bits
/*
Function: equal

Compares two bitvectors

Preconditions:

Both bitvectors must be the same length

Returns:

True if both bitvectors contain identical elements
*/
fn equal(v0: t, v1: t) -> bool {
    // FIXME: when we can break or return from inside an iterator loop,
    //        we can eliminate this painful while-loop

    let len = vec::len(v1.storage);
    let i = 0u;
    while i < len {
        if v0.storage[i] != v1.storage[i] { ret false; }
        i = i + 1u;
    }
    ret true;
}

/*
Function: clear

Set all bits to 0
*/
fn clear(v: t) {
    uint::range(0u, vec::len(v.storage)) {|i| v.storage[i] = 0u; };
}

/*
Function: set_all

Set all bits to 1
*/
fn set_all(v: t) {
    uint::range(0u, v.nbits) {|i| set(v, i, true); };
}

/*
Function: invert

Invert all bits
*/
fn invert(v: t) {
    uint::range(0u, vec::len(v.storage)) {|i|
        v.storage[i] = !v.storage[i];
    };
}

/*
Function: difference

Calculate the difference between two bitvectors

Sets each element of `v0` to the value of that element minus the element
of `v1` at the same index.

Preconditions:

Both bitvectors must be the same length

Returns:

True if `v0` was changed
*/
fn difference(v0: t, v1: t) -> bool {
    invert(v1);
    let b = intersect(v0, v1);
    invert(v1);
    ret b;
}

/*
Function: set

Set the value of a bit at a given index

Preconditions:

`i` must be less than the length of the bitvector
*/
fn set(v: t, i: uint, x: bool) {
    assert (i < v.nbits);
    let bits = uint_bits;
    let w = i / bits;
    let b = i % bits;
    let flag = 1u << b;
    v.storage[w] = if x { v.storage[w] | flag } else { v.storage[w] & !flag };
}


/*
Function: is_true

Returns true if all bits are 1
*/
fn is_true(v: t) -> bool {
    for i: uint in to_vec(v) { if i != 1u { ret false; } }
    ret true;
}


/*
Function: is_false

Returns true if all bits are 0
*/
fn is_false(v: t) -> bool {
    for i: uint in to_vec(v) { if i == 1u { ret false; } }
    ret true;
}

fn init_to_vec(v: t, i: uint) -> uint { ret if get(v, i) { 1u } else { 0u }; }

/*
Function: to_vec

Converts the bitvector to a vector of uint with the same length. Each uint
in the resulting vector has either value 0u or 1u.
*/
fn to_vec(v: t) -> [uint] {
    let sub = bind init_to_vec(v, _);
    ret vec::init_fn::<uint>(sub, v.nbits);
}

/*
Function: to_str

Converts the bitvector to a string. The resulting string has the same
length as the bitvector, and each character is either '0' or '1'.
*/
fn to_str(v: t) -> str {
    let rs = "";
    for i: uint in to_vec(v) { if i == 1u { rs += "1"; } else { rs += "0"; } }
    ret rs;
}

/*
Function: eq_vec

Compare a bitvector to a vector of uint. The uint vector is expected to
only contain the values 0u and 1u.

Preconditions:

Both the bitvector and vector must have the same length
*/
fn eq_vec(v0: t, v1: [uint]) -> bool {
    assert (v0.nbits == vec::len::<uint>(v1));
    let len = v0.nbits;
    let i = 0u;
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
    fn test_0_elements() {
        let act;
        let exp;
        act = create(0u, false);
        exp = vec::init_elt::<uint>(0u, 0u);
        assert (eq_vec(act, exp));
    }

    #[test]
    fn test_1_element() {
        let act;
        act = create(1u, false);
        assert (eq_vec(act, [0u]));
        act = create(1u, true);
        assert (eq_vec(act, [1u]));
    }

    #[test]
    fn test_10_elements() {
        let act;
        // all 0

        act = create(10u, false);
        assert (eq_vec(act, [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = create(10u, true);
        assert (eq_vec(act, [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = create(10u, false);
        set(act, 0u, true);
        set(act, 1u, true);
        set(act, 2u, true);
        set(act, 3u, true);
        set(act, 4u, true);
        assert (eq_vec(act, [1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u]));
        // mixed

        act = create(10u, false);
        set(act, 5u, true);
        set(act, 6u, true);
        set(act, 7u, true);
        set(act, 8u, true);
        set(act, 9u, true);
        assert (eq_vec(act, [0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = create(10u, false);
        set(act, 0u, true);
        set(act, 3u, true);
        set(act, 6u, true);
        set(act, 9u, true);
        assert (eq_vec(act, [1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 1u]));
    }

    #[test]
    fn test_31_elements() {
        let act;
        // all 0

        act = create(31u, false);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = create(31u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = create(31u, false);
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

        act = create(31u, false);
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

        act = create(31u, false);
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

        act = create(31u, false);
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
        let act;
        // all 0

        act = create(32u, false);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = create(32u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = create(32u, false);
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

        act = create(32u, false);
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

        act = create(32u, false);
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

        act = create(32u, false);
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
        let act;
        // all 0

        act = create(33u, false);
        assert (eq_vec(act,
                       [0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u]));
        // all 1

        act = create(33u, true);
        assert (eq_vec(act,
                       [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                        1u, 1u, 1u, 1u, 1u, 1u, 1u]));
        // mixed

        act = create(33u, false);
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

        act = create(33u, false);
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

        act = create(33u, false);
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

        act = create(33u, false);
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
