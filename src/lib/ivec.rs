// Interior vector utility functions.

import option::none;
import option::some;
import uint::next_power_of_two;
import ptr::addr_of;

type operator2[T,U,V] = fn(&T, &U) -> V;

native "rust-intrinsic" mod rusti {
    fn ivec_len[T](&T[] v) -> uint;
}

native "rust" mod rustrt {
    fn ivec_reserve_shared[T](&mutable T[mutable?] v, uint n);
    fn ivec_on_heap[T](&T[] v) -> uint;
    fn ivec_to_ptr[T](&T[] v) -> *T;
    fn ivec_copy_from_buf_shared[T](&mutable T[mutable?] v,
                                    *T ptr, uint count);
}

/// Reserves space for `n` elements in the given vector.
fn reserve[T](&mutable T[mutable?] v, uint n) {
    rustrt::ivec_reserve_shared(v, n);
}

fn on_heap[T](&T[] v) -> bool {
    ret rustrt::ivec_on_heap(v) != 0u;
}

fn to_ptr[T](&T[] v) -> *T {
    ret rustrt::ivec_to_ptr(v);
}

fn len[T](&T[mutable?] v) -> uint {
    ret rusti::ivec_len(v);
}

type init_op[T] = fn(uint) -> T;

fn init_fn[T](&init_op[T] op, uint n_elts) -> T[] {
    auto v = ~[];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[op(i)]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_fn_mut[T](&init_op[T] op, uint n_elts) -> T[mutable] {
    auto v = ~[mutable];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[mutable op(i)]; i += 1u; }
    ret v;
}

fn init_elt[T](&T t, uint n_elts) -> T[] {
    auto v = ~[];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[t]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_elt_mut[T](&T t, uint n_elts) -> T[mutable] {
    auto v = ~[mutable];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[mutable t]; i += 1u; }
    ret v;
}


// Accessors

/// Returns the last element of `v`.
fn last[T](&T[mutable?] v) -> option::t[T] {
    if (len(v) == 0u) { ret none; }
    ret some(v.(len(v) - 1u));
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
fn slice[T](&T[mutable?] v, uint start, uint end) -> T[] {
    assert (start <= end);
    assert (end <= len(v));
    auto result = ~[];
    reserve(result, end - start);
    auto i = start;
    while (i < end) { result += ~[v.(i)]; i += 1u; }
    ret result;
}

// TODO: Remove me once we have slots.
fn slice_mut[T](&T[mutable?] v, uint start, uint end) -> T[mutable] {
    assert (start <= end);
    assert (end <= len(v));
    auto result = ~[mutable];
    reserve(result, end - start);
    auto i = start;
    while (i < end) { result += ~[mutable v.(i)]; i += 1u; }
    ret result;
}


// Mutators

// TODO: Write this, unsafely, in a way that's not O(n).
fn pop[T](&mutable T[mutable?] v) -> T {
    auto ln = len(v);
    assert (ln > 0u);
    ln -= 1u;
    auto e = v.(ln);
    v = slice(v, 0u, ln);
    ret e;
}

// TODO: More.


// Appending

/// Expands the given vector in-place by appending `n` copies of `initval`.
fn grow[T](&mutable T[] v, uint n, &T initval) {
    reserve(v, next_power_of_two(len(v) + n));
    let uint i = 0u;
    while (i < n) {
        v += ~[initval];
        i += 1u;
    }
}

// TODO: Remove me once we have slots.
fn grow_mut[T](&mutable T[mutable] v, uint n, &T initval) {
    reserve(v, next_power_of_two(len(v) + n));
    let uint i = 0u;
    while (i < n) {
        v += ~[mutable initval];
        i += 1u;
    }
}

/// Calls `f` `n` times and appends the results of these calls to the given
/// vector.
fn grow_fn[T](&mutable T[] v, uint n, fn(uint)->T init_fn) {
    reserve(v, next_power_of_two(len(v) + n));
    let uint i = 0u;
    while (i < n) {
        v += ~[init_fn(i)];
        i += 1u;
    }
}

/// Sets the element at position `index` to `val`. If `index` is past the end
/// of the vector, expands the vector by replicating `initval` to fill the
/// intervening space.
fn grow_set[T](&mutable T[mutable] v, uint index, &T initval, &T val) {
    if (index >= len(v)) { grow_mut(v, index - len(v) + 1u, initval); }
    v.(index) = val;
}


// Functional utilities

fn map[T,U](fn(&T)->U f, &T[mutable?] v) -> U[] {
    auto result = ~[];
    reserve(result, len(v));
    for (T elem in v) {
        auto elem2 = elem;  // satisfies alias checker
        result += ~[f(elem2)];
    }
    ret result;
}

fn filter_map[T,U](fn(&T)->option::t[U] f, &T[mutable?] v) -> U[] {
    auto result = ~[];
    for (T elem in v) {
        auto elem2 = elem;  // satisfies alias checker
        alt (f(elem2)) {
          case (none) { /* no-op */ }
          case (some(?result_elem)) { result += ~[result_elem]; }
        }
    }
    ret result;
}

fn foldl[T,U](fn(&U,&T)->U p, &U z, &T[mutable?] v) -> U {
    auto sz = len(v);
    if (sz == 0u) { ret z; }
    auto first = v.(0);
    auto rest = slice(v, 1u, sz);
    ret p(foldl[T,U](p, z, rest), first);
}

fn any[T](fn(&T)->bool f, &T[] v) -> bool {
    for (T elem in v) { if (f(elem)) { ret true; } }
    ret false;
}

fn all[T](fn(&T)->bool f, &T[] v) -> bool {
    for (T elem in v) { if (!f(elem)) { ret false; } }
    ret true;
}

fn member[T](&T x, &T[] v) -> bool {
    for (T elt in v) { if (x == elt) { ret true; } }
    ret false;
}

fn find[T](fn(&T) -> bool  f, &T[] v) -> option::t[T] {
    for (T elt in v) { if (f(elt)) { ret some[T](elt); } }
    ret none[T];
}

mod unsafe {
    type ivec_repr = rec(mutable uint fill,
                         mutable uint alloc,
                         *mutable ivec_heap_part heap_part);
    type ivec_heap_part = rec(mutable uint fill);

    fn copy_from_buf[T](&mutable T[] v, *T ptr, uint count) {
        ret rustrt::ivec_copy_from_buf_shared(v, ptr, count);
    }

    fn from_buf[T](*T ptr, uint bytes) -> T[] {
        auto v = ~[];
        copy_from_buf(v, ptr, bytes);
        ret v;
    }

    fn set_len[T](&mutable T[] v, uint new_len) {
        auto new_fill = new_len * sys::size_of[T]();
        let *mutable ivec_repr stack_part =
            ::unsafe::reinterpret_cast(addr_of(v));
        if ((*stack_part).fill == 0u) {
            (*(*stack_part).heap_part).fill = new_fill;     // On heap.
        } else {
            (*stack_part).fill = new_fill;                  // On stack.
        }
    }
}

