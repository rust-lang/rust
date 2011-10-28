/*
Module: vec
*/

import option::{some, none};
import uint::next_power_of_two;
import ptr::addr_of;

native "rust-intrinsic" mod rusti {
    fn vec_len<T>(&&v: [mutable? T]) -> uint;
}

native "c-stack-cdecl" mod rustrt {
    fn vec_reserve_shared<T>(t: *sys::type_desc,
                             &v: [mutable? T],
                             n: uint);
    fn vec_from_buf_shared<T>(t: *sys::type_desc,
                              ptr: *T,
                              count: uint) -> [T];
}

/*
Type: init_op

A function used to initialize the elements of a vector.
*/
type init_op<T> = block(uint) -> T;


/*
Predicate: is_empty

Returns true if a vector contains no elements.
*/
pure fn is_empty<T>(v: [mutable? T]) -> bool {
    // FIXME: This would be easier if we could just call len
    for t: T in v { ret false; }
    ret true;
}

/*
Predicate: is_not_empty

Returns true if a vector contains some elements.
*/
pure fn is_not_empty<T>(v: [mutable? T]) -> bool { ret !is_empty(v); }

/*
Predicate: same_length

Returns true if two vectors have the same length
*/
pure fn same_length<T, U>(xs: [T], ys: [U]) -> bool {
    vec::len(xs) == vec::len(ys)
}

/*
Function: reserve

Reserves capacity for `n` elements in the given vector.

If the capacity for `v` is already equal to or greater than the requested
capacity, then no action is taken.

Parameters:

v - A vector
n - The number of elements to reserve space for
*/
fn reserve<T>(&v: [mutable? T], n: uint) {
    rustrt::vec_reserve_shared(sys::get_type_desc::<T>(), v, n);
}

/*
Function: len

Returns the length of a vector
*/
pure fn len<T>(v: [mutable? T]) -> uint { unchecked { rusti::vec_len(v) } }

/*
Function: init_fn

Creates and initializes an immutable vector.

Creates an immutable vector of size `n_elts` and initializes the elements
to the value returned by the function `op`.
*/
fn init_fn<T>(op: init_op<T>, n_elts: uint) -> [T] {
    let v = [];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [op(i)]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
/*
Function: init_fn

Creates and initializes a mutable vector.

Creates a mutable vector of size `n_elts` and initializes the elements to
the value returned by the function `op`.
*/
fn init_fn_mut<T>(op: init_op<T>, n_elts: uint) -> [mutable T] {
    let v = [mutable];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [mutable op(i)]; i += 1u; }
    ret v;
}

/*
Function: init_elt

Creates and initializes an immutable vector.

Creates an immutable vector of size `n_elts` and initializes the elements
to the value `t`.
*/
fn init_elt<T>(t: T, n_elts: uint) -> [T] {
    let v = [];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [t]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
/*
Function: init_elt_mut

Creates and initializes a mutable vector.

Creates a mutable vector of size `n_elts` and initializes the elements
to the value `t`.
*/
fn init_elt_mut<T>(t: T, n_elts: uint) -> [mutable T] {
    let v = [mutable];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [mutable t]; i += 1u; }
    ret v;
}

// FIXME: Possible typestate postcondition:
// len(result) == len(v) (needs issue #586)
/*
Function: to_mut

Produces a mutable vector from an immutable vector.
*/
fn to_mut<T>(v: [T]) -> [mutable T] {
    let vres = [mutable];
    for t: T in v { vres += [mutable t]; }
    ret vres;
}

// Same comment as from_mut
/*
Function: from_mut

Produces an immutable vector from a mutable vector.
*/
fn from_mut<T>(v: [mutable T]) -> [T] {
    let vres = [];
    for t: T in v { vres += [t]; }
    ret vres;
}

// Accessors

/*
Function: head

Returns the first element of a vector

Predicates:
<is_not_empty> (v)
*/
fn head<T>(v: [mutable? T]) : is_not_empty(v) -> T { ret v[0]; }

/*
Function: tail

Returns all but the first element of a vector

Predicates:
<is_not_empty> (v)
*/
fn tail<T>(v: [mutable? T]) : is_not_empty(v) -> [T] {
    ret slice(v, 1u, len(v));
}

/*
Function: last

Returns the last element of `v`

Returns:

An option containing the last element of `v` if `v` is not empty, or
none if `v` is empty.
*/
fn last<T>(v: [mutable? T]) -> option::t<T> {
    if len(v) == 0u { ret none; }
    ret some(v[len(v) - 1u]);
}

/*
Function: last_total

Returns the last element of a non-empty vector `v`

Predicates:
<is_not_empty> (v)
*/
fn last_total<T>(v: [mutable? T]) : is_not_empty(v) -> T {
    ret v[len(v) - 1u];
}

/*
Function: slice

Returns a copy of the elements from [`start`..`end`) from `v`.
*/
fn slice<T>(v: [mutable? T], start: uint, end: uint) -> [T] {
    assert (start <= end);
    assert (end <= len(v));
    let result = [];
    reserve(result, end - start);
    let i = start;
    while i < end { result += [v[i]]; i += 1u; }
    ret result;
}

// TODO: Remove me once we have slots.
/*
Function: slice_mut

Returns a copy of the elements from [`start`..`end`) from `v`.
*/
fn slice_mut<T>(v: [mutable? T], start: uint, end: uint) -> [mutable T] {
    assert (start <= end);
    assert (end <= len(v));
    let result = [mutable];
    reserve(result, end - start);
    let i = start;
    while i < end { result += [mutable v[i]]; i += 1u; }
    ret result;
}


// Mutators

/*
Function: shift

Removes the first element from a vector and return it
*/
fn shift<T>(&v: [mutable? T]) -> T {
    let ln = len::<T>(v);
    assert (ln > 0u);
    let e = v[0];
    v = slice::<T>(v, 1u, ln);
    ret e;
}

// TODO: Write this, unsafely, in a way that's not O(n).
/*
Function: pop

Remove the last element from a vector and return it
*/
fn pop<T>(&v: [mutable? T]) -> T {
    let ln = len(v);
    assert (ln > 0u);
    ln -= 1u;
    let e = v[ln];
    v = slice(v, 0u, ln);
    ret e;
}

// TODO: More.


// Appending

/*
Function: grow

Expands a vector in place, initializing the new elements to a given value

Parameters:

v - The vector to grow
n - The number of elements to add
initval - The value for the new elements
*/
fn grow<T>(&v: [T], n: uint, initval: T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += [initval]; i += 1u; }
}

// TODO: Remove me once we have slots.
// FIXME: Can't grow take a [mutable? T]
/*
Function: grow_mut

Expands a vector in place, initializing the new elements to a given value

Parameters:

v - The vector to grow
n - The number of elements to add
initval - The value for the new elements
*/
fn grow_mut<T>(&v: [mutable T], n: uint, initval: T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += [mutable initval]; i += 1u; }
}

/*
Function: grow_fn

Expands a vector in place, initializing the new elements to the result of a
function

Function `init_fn` is called `n` times with the values [0..`n`)

Parameters:

v - The vector to grow
n - The number of elements to add
init_fn - A function to call to retreive each appended element's value
*/
fn grow_fn<T>(&v: [T], n: uint, op: init_op<T>) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += [op(i)]; i += 1u; }
}

/*
Function: grow_set

Sets the value of a vector element at a given index, growing the vector as
needed

Sets the element at position `index` to `val`. If `index` is past the end
of the vector, expands the vector by replicating `initval` to fill the
intervening space.
*/
fn grow_set<T>(&v: [mutable T], index: uint, initval: T, val: T) {
    if index >= len(v) { grow_mut(v, index - len(v) + 1u, initval); }
    v[index] = val;
}


// Functional utilities

/*
Function: map

Apply a function to each element of a vector and return the results
*/
fn map<T, U>(f: block(T) -> U, v: [mutable? T]) -> [U] {
    let result = [];
    reserve(result, len(v));
    for elem: T in v {
        let elem2 = elem; // satisfies alias checker
        result += [f(elem2)];
    }
    ret result;
}

/*
Function: map2

Apply a function to each pair of elements and return the results
*/
fn map2<T, U, V>(f: block(T, U) -> V, v0: [T], v1: [U]) -> [V] {
    let v0_len = len::<T>(v0);
    if v0_len != len::<U>(v1) { fail; }
    let u: [V] = [];
    let i = 0u;
    while i < v0_len { u += [f({ v0[i] }, { v1[i] })]; i += 1u; }
    ret u;
}

/*
Function: filter_map

Apply a function to each element of a vector and return the results

If function `f` returns `none` then that element is excluded from
the resulting vector.
*/
fn filter_map<T, U>(f: block(T) -> option::t<U>, v: [mutable? T]) -> [U] {
    let result = [];
    for elem: T in v {
        let elem2 = elem; // satisfies alias checker
        alt f(elem2) {
          none. {/* no-op */ }
          some(result_elem) { result += [result_elem]; }
        }
    }
    ret result;
}

/*
Function: filter

Construct a new vector from the elements of a vector for which some predicate
holds.

Apply function `f` to each element of `v` and return a vector containing
only those elements for which `f` returned true.
*/
fn filter<T>(f: block(T) -> bool, v: [mutable? T]) -> [T] {
    let result = [];
    for elem: T in v {
        let elem2 = elem; // satisfies alias checker
        if f(elem2) {
            result += [elem2];
        }
    }
    ret result;
}

/*
Function: foldl

Reduce a vector from left to right
*/
fn foldl<T, U>(p: block(T, U) -> T, z: T, v: [mutable? U]) -> T {
    let accum = z;
    iter(v) { |elt|
        accum = p(accum, elt);
    }
    ret accum;
}

/*
Function: foldr

Reduce a vector from right to left
*/
fn foldr<T, U>(p: block(T, U) -> U, z: U, v: [mutable? T]) -> U {
    let accum = z;
    riter(v) { |elt|
        accum = p(elt, accum);
    }
    ret accum;
}

/*
Function: any

Return true if a predicate matches any elements

If the vector contains no elements then false is returned.
*/
fn any<T>(f: block(T) -> bool, v: [T]) -> bool {
    for elem: T in v { if f(elem) { ret true; } }
    ret false;
}

/*
Function: all

Return true if a predicate matches all elements

If the vector contains no elements then true is returned.
*/
fn all<T>(f: block(T) -> bool, v: [T]) -> bool {
    for elem: T in v { if !f(elem) { ret false; } }
    ret true;
}

/*
Function: member

Return true if a vector contains an element with the given value
*/
fn member<T>(x: T, v: [T]) -> bool {
    for elt: T in v { if x == elt { ret true; } }
    ret false;
}

/*
Function: count

Returns the number of elements that are equal to a given value
*/
fn count<T>(x: T, v: [mutable? T]) -> uint {
    let cnt = 0u;
    for elt: T in v { if x == elt { cnt += 1u; } }
    ret cnt;
}

/*
Function: find

Search for an element that matches a given predicate

Apply function `f` to each element of `v`, starting from the first.
When function `f` returns true then an option containing the element
is returned. If `f` matches no elements then none is returned.
*/
fn find<T>(f: block(T) -> bool, v: [T]) -> option::t<T> {
    for elt: T in v { if f(elt) { ret some(elt); } }
    ret none;
}

/*
Function: position

Find the first index containing a matching value

Returns:

option::some(uint) - The first index containing a matching value
option::none - No elements matched
*/
fn position<T>(x: T, v: [T]) -> option::t<uint> {
    let i: uint = 0u;
    while i < len(v) { if x == v[i] { ret some::<uint>(i); } i += 1u; }
    ret none;
}

/*
Function: position_pred

Find the first index for which the value matches some predicate
*/
fn position_pred<T>(f: block(T) -> bool, v: [T]) -> option::t<uint> {
    let i: uint = 0u;
    while i < len(v) { if f(v[i]) { ret some::<uint>(i); } i += 1u; }
    ret none;
}

// FIXME: if issue #586 gets implemented, could have a postcondition
// saying the two result lists have the same length -- or, could
// return a nominal record with a constraint saying that, instead of
// returning a tuple (contingent on issue #869)
/*
Function: unzip

Convert a vector of pairs into a pair of vectors

Returns a tuple containing two vectors where the i-th element of the first
vector contains the first element of the i-th tuple of the input vector,
and the i-th element of the second vector contains the second element
of the i-th tuple of the input vector.
*/
fn unzip<T, U>(v: [(T, U)]) -> ([T], [U]) {
    let as = [], bs = [];
    for (a, b) in v { as += [a]; bs += [b]; }
    ret (as, bs);
}

/*
Function: zip

Convert two vectors to a vector of pairs

Returns a vector of tuples, where the i-th tuple contains contains the
i-th elements from each of the input vectors.

Preconditions:

<same_length> (v, u)
*/
fn zip<T, U>(v: [T], u: [U]) : same_length(v, u) -> [(T, U)] {
    let zipped = [];
    let sz = len(v), i = 0u;
    assert (sz == len(u));
    while i < sz { zipped += [(v[i], u[i])]; i += 1u; }
    ret zipped;
}

/*
Function: swap

Swaps two elements in a vector

Parameters:
v - The input vector
a - The index of the first element
b - The index of the second element
*/
fn swap<T>(v: [mutable T], a: uint, b: uint) {
    let t: T = v[a];
    v[a] = v[b];
    v[b] = t;
}

/*
Function: reverse

Reverse the order of elements in a vector, in place
*/
fn reverse<T>(v: [mutable T]) {
    let i: uint = 0u;
    let ln = len::<T>(v);
    while i < ln / 2u { swap(v, i, ln - i - 1u); i += 1u; }
}


/*
Function: reversed

Returns a vector with the order of elements reversed
*/
fn reversed<T>(v: [mutable? T]) -> [T] {
    let rs: [T] = [];
    let i = len::<T>(v);
    if i == 0u { ret rs; } else { i -= 1u; }
    while i != 0u { rs += [v[i]]; i -= 1u; }
    rs += [v[0]];
    ret rs;
}

// FIXME: Seems like this should take char params. Maybe belongs in char
/*
Function: enum_chars

Returns a vector containing a range of chars
*/
fn enum_chars(start: u8, end: u8) : u8::le(start, end) -> [char] {
    let i = start;
    let r = [];
    while i <= end { r += [i as char]; i += 1u as u8; }
    ret r;
}

// FIXME: Probably belongs in uint. Compare to uint::range
/*
Function: enum_uints

Returns a vector containing a range of uints
*/
fn enum_uints(start: uint, end: uint) : uint::le(start, end) -> [uint] {
    let i = start;
    let r = [];
    while i <= end { r += [i]; i += 1u; }
    ret r;
}

/*
Function: iter

Iterates over a vector

Iterates over vector `v` and, for each element, calls function `f` with the
element's value.

*/
fn iter<T>(v: [mutable? T], f: block(T)) {
    iter2(v) { |_i, v| f(v) }
}

/*
Function: iter2

Iterates over a vector's elements and indexes

Iterates over vector `v` and, for each element, calls function `f` with the
element's value and index.
*/
fn iter2<T>(v: [mutable? T], f: block(uint, T)) {
    let i = 0u;
    for x in v { f(i, x); i += 1u; }
}

/*
Function: riter

Iterates over a vector in reverse

Iterates over vector `v` and, for each element, calls function `f` with the
element's value.

*/
fn riter<T>(v: [mutable? T], f: block(T)) {
    riter2(v) { |_i, v| f(v) }
}

/*
Function: riter2

Iterates over a vector's elements and indexes in reverse

Iterates over vector `v` and, for each element, calls function `f` with the
element's value and index.
*/
fn riter2<T>(v: [mutable? T], f: block(uint, T)) {
    let i = len(v);
    while 0u < i {
        i -= 1u;
        f(i, v[i]);
    };
}

/*
Function: to_ptr

FIXME: We don't need this wrapper
*/
unsafe fn to_ptr<T>(v: [T]) -> *T { ret unsafe::to_ptr(v); }

/*
Module: unsafe
*/
mod unsafe {
    type vec_repr = {mutable fill: uint, mutable alloc: uint, data: u8};

    /*
    Function: from_buf

    Constructs a vector from an unsafe pointer to a buffer

    Parameters:

    ptr - An unsafe pointer to a buffer of `T`
    elts - The number of elements in the buffer
    */
    unsafe fn from_buf<T>(ptr: *T, elts: uint) -> [T] {
        ret rustrt::vec_from_buf_shared(sys::get_type_desc::<T>(),
                                        ptr, elts);
    }

    /*
    Function: set_len

    Sets the length of a vector

    This well explicitly set the size of the vector, without actually
    modifing its buffers, so it is up to the caller to ensure that
    the vector is actually the specified size.
    */
    unsafe fn set_len<T>(&v: [T], new_len: uint) {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).fill = new_len * sys::size_of::<T>();
    }

    /*
    Function: to_ptr

    Returns an unsafe pointer to the vector's buffer

    The caller must ensure that the vector outlives the pointer this
    function returns, or else it will end up pointing to garbage.

    Modifying the vector may cause its buffer to be reallocated, which
    would also make any pointers to it invalid.
    */
    unsafe fn to_ptr<T>(v: [T]) -> *T {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        ret ::unsafe::reinterpret_cast(addr_of((**repr).data));
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
