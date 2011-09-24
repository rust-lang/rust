// Interior vector utility functions.

import option::{some, none};
import uint::next_power_of_two;
import ptr::addr_of;

native "rust-intrinsic" mod rusti {
    fn vec_len<T>(v: [T]) -> uint;
}

native "rust" mod rustrt {
    fn vec_reserve_shared<T>(&v: [mutable? T], n: uint);
    fn vec_from_buf_shared<T>(ptr: *T, count: uint) -> [T];
}

/// Reserves space for `n` elements in the given vector.
fn reserve<@T>(&v: [mutable? T], n: uint) {
    rustrt::vec_reserve_shared(v, n);
}

fn len<T>(v: [mutable? T]) -> uint { ret rusti::vec_len(v); }

type init_op<T> = fn(uint) -> T;

fn init_fn<@T>(op: init_op<T>, n_elts: uint) -> [T] {
    let v = [];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [op(i)]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_fn_mut<@T>(op: init_op<T>, n_elts: uint) -> [mutable T] {
    let v = [mutable];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [mutable op(i)]; i += 1u; }
    ret v;
}

fn init_elt<@T>(t: T, n_elts: uint) -> [T] {
    let v = [];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [t]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_elt_mut<@T>(t: T, n_elts: uint) -> [mutable T] {
    let v = [mutable];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [mutable t]; i += 1u; }
    ret v;
}

// FIXME: Possible typestate postcondition:
// len(result) == len(v) (needs issue #586)
fn to_mut<@T>(v: [T]) -> [mutable T] {
    let vres = [mutable];
    for t: T in v { vres += [mutable t]; }
    ret vres;
}

// Same comment as from_mut
fn from_mut<@T>(v: [mutable T]) -> [T] {
    let vres = [];
    for t: T in v { vres += [t]; }
    ret vres;
}

// Predicates
pure fn is_empty<T>(v: [mutable? T]) -> bool {
    // FIXME: This would be easier if we could just call len
    for t: T in v { ret false; }
    ret true;
}

pure fn is_not_empty<T>(v: [mutable? T]) -> bool { ret !is_empty(v); }

// Accessors

/// Returns the first element of a vector
fn head<@T>(v: [mutable? T]) : is_not_empty(v) -> T { ret v[0]; }

/// Returns all but the first element of a vector
fn tail<@T>(v: [mutable? T]) : is_not_empty(v) -> [mutable? T] {
    ret slice(v, 1u, len(v));
}

/// Returns the last element of `v`.
fn last<@T>(v: [mutable? T]) -> option::t<T> {
    if len(v) == 0u { ret none; }
    ret some(v[len(v) - 1u]);
}

/// Returns the last element of a non-empty vector `v`.
fn last_total<@T>(v: [mutable? T]) : is_not_empty(v) -> T {
    ret v[len(v) - 1u];
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
fn slice<@T>(v: [mutable? T], start: uint, end: uint) -> [T] {
    assert (start <= end);
    assert (end <= len(v));
    let result = [];
    reserve(result, end - start);
    let i = start;
    while i < end { result += [v[i]]; i += 1u; }
    ret result;
}

// TODO: Remove me once we have slots.
fn slice_mut<@T>(v: [mutable? T], start: uint, end: uint) -> [mutable T] {
    assert (start <= end);
    assert (end <= len(v));
    let result = [mutable];
    reserve(result, end - start);
    let i = start;
    while i < end { result += [mutable v[i]]; i += 1u; }
    ret result;
}


// Mutators

fn shift<@T>(&v: [mutable? T]) -> T {
    let ln = len::<T>(v);
    assert (ln > 0u);
    let e = v[0];
    v = slice::<T>(v, 1u, ln);
    ret e;
}

// TODO: Write this, unsafely, in a way that's not O(n).
fn pop<@T>(&v: [mutable? T]) -> T {
    let ln = len(v);
    assert (ln > 0u);
    ln -= 1u;
    let e = v[ln];
    v = slice(v, 0u, ln);
    ret e;
}

// TODO: More.


// Appending

/// Expands the given vector in-place by appending `n` copies of `initval`.
fn grow<@T>(&v: [T], n: uint, initval: T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += [initval]; i += 1u; }
}

// TODO: Remove me once we have slots.
fn grow_mut<@T>(&v: [mutable T], n: uint, initval: T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += [mutable initval]; i += 1u; }
}

/// Calls `f` `n` times and appends the results of these calls to the given
/// vector.
fn grow_fn<@T>(&v: [T], n: uint, init_fn: fn(uint) -> T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += [init_fn(i)]; i += 1u; }
}

/// Sets the element at position `index` to `val`. If `index` is past the end
/// of the vector, expands the vector by replicating `initval` to fill the
/// intervening space.
fn grow_set<@T>(&v: [mutable T], index: uint, initval: T, val: T) {
    if index >= len(v) { grow_mut(v, index - len(v) + 1u, initval); }
    v[index] = val;
}


// Functional utilities

fn map<@T, @U>(f: block(T) -> U, v: [mutable? T]) -> [U] {
    let result = [];
    reserve(result, len(v));
    for elem: T in v {
        let elem2 = elem; // satisfies alias checker
        result += [f(elem2)];
    }
    ret result;
}

fn map2<@T, @U, @V>(f: block(T, U) -> V, v0: [T], v1: [U]) -> [V] {
    let v0_len = len::<T>(v0);
    if v0_len != len::<U>(v1) { fail; }
    let u: [V] = [];
    let i = 0u;
    while i < v0_len { u += [f({ v0[i] }, { v1[i] })]; i += 1u; }
    ret u;
}

fn filter_map<@T, @U>(f: block(T) -> option::t<U>, v: [mutable? T]) -> [U] {
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

fn filter<@T>(f: block(T) -> bool, v: [mutable? T]) -> [T] {
    let result = [];
    for elem: T in v {
        let elem2 = elem; // satisfies alias checker
        if f(elem2) {
            result += [elem2];
        }
    }
    ret result;
}

fn foldl<@T, @U>(p: block(U, T) -> U, z: U, v: [mutable? T]) -> U {
    let sz = len(v);
    if sz == 0u { ret z; }
    let first = v[0];
    let rest = slice(v, 1u, sz);
    ret p(foldl(p, z, rest), first);
}

fn any<T>(f: block(T) -> bool, v: [T]) -> bool {
    for elem: T in v { if f(elem) { ret true; } }
    ret false;
}

fn all<T>(f: block(T) -> bool, v: [T]) -> bool {
    for elem: T in v { if !f(elem) { ret false; } }
    ret true;
}

fn member<T>(x: T, v: [T]) -> bool {
    for elt: T in v { if x == elt { ret true; } }
    ret false;
}

fn count<T>(x: T, v: [mutable? T]) -> uint {
    let cnt = 0u;
    for elt: T in v { if x == elt { cnt += 1u; } }
    ret cnt;
}

fn find<@T>(f: block(T) -> bool, v: [T]) -> option::t<T> {
    for elt: T in v { if f(elt) { ret some(elt); } }
    ret none;
}

fn position<@T>(x: T, v: [T]) -> option::t<uint> {
    let i: uint = 0u;
    while i < len(v) { if x == v[i] { ret some::<uint>(i); } i += 1u; }
    ret none;
}

fn position_pred<T>(f: fn(T) -> bool, v: [T]) -> option::t<uint> {
    let i: uint = 0u;
    while i < len(v) { if f(v[i]) { ret some::<uint>(i); } i += 1u; }
    ret none;
}

pure fn same_length<T, U>(xs: [T], ys: [U]) -> bool {
    let xlen = unchecked{ vec::len(xs) };
    let ylen = unchecked{ vec::len(ys) };
    xlen == ylen
}

// FIXME: if issue #586 gets implemented, could have a postcondition
// saying the two result lists have the same length -- or, could
// return a nominal record with a constraint saying that, instead of
// returning a tuple (contingent on issue #869)
fn unzip<@T, @U>(v: [(T, U)]) -> ([T], [U]) {
    let as = [], bs = [];
    for (a, b) in v { as += [a]; bs += [b]; }
    ret (as, bs);
}

fn zip<@T, @U>(v: [T], u: [U]) : same_length(v, u) -> [(T, U)] {
    let zipped = [];
    let sz = len(v), i = 0u;
    assert (sz == len(u));
    while i < sz { zipped += [(v[i], u[i])]; i += 1u; }
    ret zipped;
}

// Swaps two elements in a vector
fn swap<@T>(v: [mutable T], a: uint, b: uint) {
    let t: T = v[a];
    v[a] = v[b];
    v[b] = t;
}

// In place vector reversal
fn reverse<@T>(v: [mutable T]) {
    let i: uint = 0u;
    let ln = len::<T>(v);
    while i < ln / 2u { swap(v, i, ln - i - 1u); i += 1u; }
}


// Functional vector reversal. Returns a reversed copy of v.
fn reversed<@T>(v: [T]) -> [T] {
    let rs: [T] = [];
    let i = len::<T>(v);
    if i == 0u { ret rs; } else { i -= 1u; }
    while i != 0u { rs += [v[i]]; i -= 1u; }
    rs += [v[0]];
    ret rs;
}

// Generating vecs.
fn enum_chars(start: u8, end: u8) : u8::le(start, end) -> [char] {
    let i = start;
    let r = [];
    while i <= end { r += [i as char]; i += 1u as u8; }
    ret r;
}

fn enum_uints(start: uint, end: uint) : uint::le(start, end) -> [uint] {
    let i = start;
    let r = [];
    while i <= end { r += [i]; i += 1u; }
    ret r;
}

// Iterate over a list with with the indexes
iter iter2<@T>(v: [T]) -> (uint, T) {
    let i = 0u;
    for x in v { put (i, x); i += 1u; }
}

mod unsafe {
    type vec_repr = {mutable fill: uint, mutable alloc: uint, data: u8};

    fn from_buf<@T>(ptr: *T, elts: uint) -> [T] {
        ret rustrt::vec_from_buf_shared(ptr, elts);
    }

    fn set_len<T>(&v: [T], new_len: uint) {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).fill = new_len * sys::size_of::<T>();
    }

    fn to_ptr<T>(v: [T]) -> *T {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        ret ::unsafe::reinterpret_cast(addr_of((**repr).data));
    }
}

fn to_ptr<T>(v: [T]) -> *T { ret unsafe::to_ptr(v); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
