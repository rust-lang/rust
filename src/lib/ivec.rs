// Interior vector utility functions.

import option::none;
import option::some;
import uint::next_power_of_two;
import ptr::addr_of;

type operator2[T, U, V] = fn(&T, &U) -> V ;

native "rust-intrinsic" mod rusti {
    fn ivec_len[T](v: &T[]) -> uint;
}

native "rust" mod rustrt {
    fn ivec_reserve_shared[T](v: &mutable T[mutable? ], n: uint);
    fn ivec_on_heap[T](v: &T[]) -> uint;
    fn ivec_to_ptr[T](v: &T[]) -> *T;
    fn ivec_copy_from_buf_shared[T](v: &mutable T[mutable? ], ptr: *T,
                                    count: uint);
}

/// Reserves space for `n` elements in the given vector.
fn reserve[T](v: &mutable T[mutable? ], n: uint) {
    rustrt::ivec_reserve_shared(v, n);
}

fn on_heap[T](v: &T[]) -> bool { ret rustrt::ivec_on_heap(v) != 0u; }

fn to_ptr[T](v: &T[]) -> *T { ret rustrt::ivec_to_ptr(v); }

fn len[T](v: &T[mutable? ]) -> uint { ret rusti::ivec_len(v); }

type init_op[T] = fn(uint) -> T ;

fn init_fn[T](op: &init_op[T], n_elts: uint) -> T[] {
    let v = ~[];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += ~[op(i)]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_fn_mut[T](op: &init_op[T], n_elts: uint) -> T[mutable ] {
    let v = ~[mutable ];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += ~[mutable op(i)]; i += 1u; }
    ret v;
}

fn init_elt[T](t: &T, n_elts: uint) -> T[] {
    let v = ~[];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += ~[t]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_elt_mut[T](t: &T, n_elts: uint) -> T[mutable ] {
    let v = ~[mutable ];
    reserve(v, n_elts);
    let i: uint = 0u;
    while i < n_elts { v += ~[mutable t]; i += 1u; }
    ret v;
}

fn to_mut[T](v: &T[]) -> T[mutable ] {
    let vres = ~[mutable ];
    for t: T  in v { vres += ~[mutable t]; }
    ret vres;
}

fn from_mut[T](v: &T[mutable ]) -> T[] {
    let vres = ~[];
    for t: T  in v { vres += ~[t]; }
    ret vres;
}

// Predicates
pred is_empty[T](v: &T[mutable? ]) -> bool {
    // FIXME: This would be easier if we could just call len
    for t: T  in v { ret false; }
    ret true;
}

pred is_not_empty[T](v: &T[mutable? ]) -> bool { ret !is_empty(v); }

// Accessors

/// Returns the first element of a vector
fn head[T](v: &T[mutable? ]) -> T { ret v.(0); }

/// Returns all but the first element of a vector
fn tail[T](v: &T[mutable? ]) -> T[mutable? ] { ret slice(v, 1u, len(v)); }

/// Returns the last element of `v`.
fn last[T](v: &T[mutable? ]) -> option::t[T] {
    if len(v) == 0u { ret none; }
    ret some(v.(len(v) - 1u));
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
fn slice[T](v: &T[mutable? ], start: uint, end: uint) -> T[] {
    assert (start <= end);
    assert (end <= len(v));
    let result = ~[];
    reserve(result, end - start);
    let i = start;
    while i < end { result += ~[v.(i)]; i += 1u; }
    ret result;
}

// TODO: Remove me once we have slots.
fn slice_mut[T](v: &T[mutable? ], start: uint, end: uint) -> T[mutable ] {
    assert (start <= end);
    assert (end <= len(v));
    let result = ~[mutable ];
    reserve(result, end - start);
    let i = start;
    while i < end { result += ~[mutable v.(i)]; i += 1u; }
    ret result;
}


// Mutators

// TODO: Write this, unsafely, in a way that's not O(n).
fn pop[T](v: &mutable T[mutable? ]) -> T {
    let ln = len(v);
    assert (ln > 0u);
    ln -= 1u;
    let e = v.(ln);
    v = slice(v, 0u, ln);
    ret e;
}

// TODO: More.


// Appending

/// Expands the given vector in-place by appending `n` copies of `initval`.
fn grow[T](v: &mutable T[], n: uint, initval: &T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += ~[initval]; i += 1u; }
}

// TODO: Remove me once we have slots.
fn grow_mut[T](v: &mutable T[mutable ], n: uint, initval: &T) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += ~[mutable initval]; i += 1u; }
}

/// Calls `f` `n` times and appends the results of these calls to the given
/// vector.
fn grow_fn[T](v: &mutable T[], n: uint, init_fn: fn(uint) -> T ) {
    reserve(v, next_power_of_two(len(v) + n));
    let i: uint = 0u;
    while i < n { v += ~[init_fn(i)]; i += 1u; }
}

/// Sets the element at position `index` to `val`. If `index` is past the end
/// of the vector, expands the vector by replicating `initval` to fill the
/// intervening space.
fn grow_set[T](v: &mutable T[mutable ], index: uint, initval: &T, val: &T) {
    if index >= len(v) { grow_mut(v, index - len(v) + 1u, initval); }
    v.(index) = val;
}


// Functional utilities

fn map[T, U](f: fn(&T) -> U , v: &T[mutable? ]) -> U[] {
    let result = ~[];
    reserve(result, len(v));
    for elem: T  in v {
        let elem2 = elem; // satisfies alias checker
        result += ~[f(elem2)];
    }
    ret result;
}

fn filter_map[T, U](f: fn(&T) -> option::t[U] , v: &T[mutable? ]) -> U[] {
    let result = ~[];
    for elem: T  in v {
        let elem2 = elem; // satisfies alias checker
        alt f(elem2) {
          none. {/* no-op */ }
          some(result_elem) { result += ~[result_elem]; }
        }
    }
    ret result;
}

fn foldl[T, U](p: fn(&U, &T) -> U , z: &U, v: &T[mutable? ]) -> U {
    let sz = len(v);
    if sz == 0u { ret z; }
    let first = v.(0);
    let rest = slice(v, 1u, sz);
    ret p(foldl[T, U](p, z, rest), first);
}

fn any[T](f: fn(&T) -> bool , v: &T[]) -> bool {
    for elem: T  in v { if f(elem) { ret true; } }
    ret false;
}

fn all[T](f: fn(&T) -> bool , v: &T[]) -> bool {
    for elem: T  in v { if !f(elem) { ret false; } }
    ret true;
}

fn member[T](x: &T, v: &T[]) -> bool {
    for elt: T  in v { if x == elt { ret true; } }
    ret false;
}

fn count[T](x: &T, v: &T[mutable? ]) -> uint {
    let cnt = 0u;
    for elt: T  in v { if x == elt { cnt += 1u; } }
    ret cnt;
}

fn find[T](f: fn(&T) -> bool , v: &T[]) -> option::t[T] {
    for elt: T  in v { if f(elt) { ret some[T](elt); } }
    ret none[T];
}

fn unzip[T, U](v: &{_0: T, _1: U}[]) -> {_0: T[], _1: U[]} {
    let sz = len(v);
    if sz == 0u {
        ret {_0: ~[], _1: ~[]};
    } else {
        let rest = slice(v, 1u, sz);
        let tl = unzip(rest);
        let a = ~[v.(0)._0];
        let b = ~[v.(0)._1];
        ret {_0: a + tl._0, _1: b + tl._1};
    }
}


// FIXME make the lengths being equal a constraint
fn zip[T, U](v: &T[], u: &U[]) -> {_0: T, _1: U}[] {
    let sz = len(v);
    assert (sz == len(u));
    if sz == 0u {
        ret ~[];
    } else {
        let rest = zip(slice(v, 1u, sz), slice(u, 1u, sz));
        ret ~[{_0: v.(0), _1: u.(0)}] + rest;
    }
}

mod unsafe {
    type ivec_repr =
        {mutable fill: uint,
         mutable alloc: uint,
         heap_part: *mutable ivec_heap_part};
    type ivec_heap_part = {mutable fill: uint};

    fn copy_from_buf[T](v: &mutable T[], ptr: *T, count: uint) {
        ret rustrt::ivec_copy_from_buf_shared(v, ptr, count);
    }

    fn from_buf[T](ptr: *T, bytes: uint) -> T[] {
        let v = ~[];
        copy_from_buf(v, ptr, bytes);
        ret v;
    }

    fn set_len[T](v: &mutable T[], new_len: uint) {
        let new_fill = new_len * sys::size_of[T]();
        let stack_part: *mutable ivec_repr =
            ::unsafe::reinterpret_cast(addr_of(v));
        if (*stack_part).fill == 0u {
            (*(*stack_part).heap_part).fill = new_fill; // On heap.
        } else {
            (*stack_part).fill = new_fill; // On stack.
        }
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
