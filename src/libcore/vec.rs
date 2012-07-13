//! Vectors

import option::{some, none};
import ptr::addr_of;
import libc::size_t;

export append;
export append_one;
export consume;
export init_op;
export is_empty;
export is_not_empty;
export same_length;
export reserve;
export reserve_at_least;
export capacity;
export len;
export from_fn;
export from_elem;
export to_mut;
export from_mut;
export head;
export tail;
export tailn;
export init;
export last;
export last_opt;
export slice;
export view;
export split;
export splitn;
export rsplit;
export rsplitn;
export shift;
export unshift;
export pop;
export push, push_all, push_all_move;
export grow;
export grow_fn;
export grow_set;
export map;
export mapi;
export map2;
export map_consume;
export flat_map;
export filter_map;
export filter;
export concat;
export connect;
export foldl;
export foldr;
export any;
export any2;
export all;
export alli;
export all2;
export contains;
export count;
export find;
export find_between;
export rfind;
export rfind_between;
export position_elem;
export position;
export position_between;
export position_elem;
export rposition;
export rposition_between;
export unzip;
export zip;
export swap;
export reverse;
export reversed;
export iter, iter_between, each, eachi;
export iter2;
export iteri;
export riter;
export riteri;
export permute;
export windowed;
export as_buf;
export as_mut_buf;
export unpack_slice;
export unpack_const_slice;
export unsafe;
export u8;
export extensions;

#[abi = "cdecl"]
extern mod rustrt {
    fn vec_reserve_shared(++t: *sys::type_desc,
                          ++v: **unsafe::vec_repr,
                          ++n: libc::size_t);
    fn vec_from_buf_shared(++t: *sys::type_desc,
                           ++ptr: *(),
                           ++count: libc::size_t) -> *unsafe::vec_repr;
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn move_val_init<T>(&dst: T, -src: T);
}

/// A function used to initialize the elements of a vector
type init_op<T> = fn(uint) -> T;

/// Returns true if a vector contains no elements
pure fn is_empty<T>(v: &[const T]) -> bool {
    unpack_const_slice(v, |_p, len| len == 0u)
}

/// Returns true if a vector contains some elements
pure fn is_not_empty<T>(v: &[const T]) -> bool {
    unpack_const_slice(v, |_p, len| len > 0u)
}

/// Returns true if two vectors have the same length
pure fn same_length<T, U>(xs: &[const T], ys: &[const U]) -> bool {
    len(xs) == len(ys)
}

/**
 * Reserves capacity for exactly `n` elements in the given vector.
 *
 * If the capacity for `v` is already equal to or greater than the requested
 * capacity, then no action is taken.
 *
 * # Arguments
 *
 * * v - A vector
 * * n - The number of elements to reserve space for
 */
fn reserve<T>(&v: ~[const T], n: uint) {
    // Only make the (slow) call into the runtime if we have to
    if capacity(v) < n {
        let ptr = ptr::addr_of(v) as **unsafe::vec_repr;
        rustrt::vec_reserve_shared(sys::get_type_desc::<T>(),
                                   ptr, n as size_t);
    }
}

/**
 * Reserves capacity for at least `n` elements in the given vector.
 *
 * This function will over-allocate in order to amortize the allocation costs
 * in scenarios where the caller may need to repeatedly reserve additional
 * space.
 *
 * If the capacity for `v` is already equal to or greater than the requested
 * capacity, then no action is taken.
 *
 * # Arguments
 *
 * * v - A vector
 * * n - The number of elements to reserve space for
 */
fn reserve_at_least<T>(&v: ~[const T], n: uint) {
    reserve(v, uint::next_power_of_two(n));
}

/// Returns the number of elements the vector can hold without reallocating
#[inline(always)]
pure fn capacity<T>(&&v: ~[const T]) -> uint {
    unsafe {
        let repr: **unsafe::vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).alloc / sys::size_of::<T>()
    }
}

/// Returns the length of a vector
#[inline(always)]
pure fn len<T>(&&v: &[const T]) -> uint {
    unpack_const_slice(v, |_p, len| len)
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pure fn from_fn<T>(n_elts: uint, op: init_op<T>) -> ~[T] {
    let mut v = ~[];
    unchecked{reserve(v, n_elts);}
    let mut i: uint = 0u;
    while i < n_elts unsafe { push(v, op(i)); i += 1u; }
    ret v;
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pure fn from_elem<T: copy>(n_elts: uint, t: T) -> ~[T] {
    let mut v = ~[];
    unchecked{reserve(v, n_elts)}
    let mut i: uint = 0u;
    unsafe { // because push is impure
        while i < n_elts { push(v, t); i += 1u; }
    }
    ret v;
}

/// Produces a mut vector from an immutable vector.
fn to_mut<T>(+v: ~[T]) -> ~[mut T] {
    unsafe { ::unsafe::transmute(v) }
}

/// Produces an immutable vector from a mut vector.
fn from_mut<T>(+v: ~[mut T]) -> ~[T] {
    unsafe { ::unsafe::transmute(v) }
}

// Accessors

/// Returns the first element of a vector
pure fn head<T: copy>(v: &[const T]) -> T { v[0] }

/// Returns a vector containing all but the first element of a slice
pure fn tail<T: copy>(v: &[const T]) -> ~[T] {
    ret slice(v, 1u, len(v));
}

/**
 * Returns a vector containing all but the first `n` \
 * elements of a slice
 */
pure fn tailn<T: copy>(v: &[const T], n: uint) -> ~[T] {
    slice(v, n, len(v))
}

/// Returns a vector containing all but the last element of a slice
pure fn init<T: copy>(v: &[const T]) -> ~[T] {
    assert len(v) != 0u;
    slice(v, 0u, len(v) - 1u)
}

/// Returns the last element of the slice `v`, failing if the slice is empty.
pure fn last<T: copy>(v: &[const T]) -> T {
    if len(v) == 0u { fail "last_unsafe: empty vector" }
    v[len(v) - 1u]
}

/**
 * Returns `some(x)` where `x` is the last element of the slice `v`,
 * or `none` if the vector is empty.
 */
pure fn last_opt<T: copy>(v: &[const T]) -> option<T> {
    if len(v) == 0u { ret none; }
    some(v[len(v) - 1u])
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
pure fn slice<T: copy>(v: &[const T], start: uint, end: uint) -> ~[T] {
    assert (start <= end);
    assert (end <= len(v));
    let mut result = ~[];
    unchecked {
        for uint::range(start, end) |i| { vec::push(result, v[i]) }
    }
    ret result;
}

#[doc = "Return a slice that points into another slice."]
pure fn view<T>(v: &a.[T], start: uint, end: uint) -> &a.[T] {
    assert (start <= end);
    assert (end <= len(v));
    do unpack_slice(v) |p, _len| {
        unsafe {
            ::unsafe::reinterpret_cast(
                (ptr::offset(p, start), (end - start) * sys::size_of::<T>()))
        }
    }
}

/// Split the vector `v` by applying each element against the predicate `f`.
fn split<T: copy>(v: &[T], f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { ret ~[] }

    let mut start = 0u;
    let mut result = ~[];
    while start < ln {
        alt position_between(v, start, ln, f) {
          none { break }
          some(i) {
            push(result, slice(v, start, i));
            start = i + 1u;
          }
        }
    }
    push(result, slice(v, start, ln));
    result
}

/**
 * Split the vector `v` by applying each element against the predicate `f` up
 * to `n` times.
 */
fn splitn<T: copy>(v: &[T], n: uint, f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { ret ~[] }

    let mut start = 0u;
    let mut count = n;
    let mut result = ~[];
    while start < ln && count > 0u {
        alt position_between(v, start, ln, f) {
          none { break }
          some(i) {
            push(result, slice(v, start, i));
            // Make sure to skip the separator.
            start = i + 1u;
            count -= 1u;
          }
        }
    }
    push(result, slice(v, start, ln));
    result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f`.
 */
fn rsplit<T: copy>(v: &[T], f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { ret ~[] }

    let mut end = ln;
    let mut result = ~[];
    while end > 0u {
        alt rposition_between(v, 0u, end, f) {
          none { break }
          some(i) {
            push(result, slice(v, i + 1u, end));
            end = i;
          }
        }
    }
    push(result, slice(v, 0u, end));
    reversed(result)
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f` up to `n times.
 */
fn rsplitn<T: copy>(v: &[T], n: uint, f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { ret ~[] }

    let mut end = ln;
    let mut count = n;
    let mut result = ~[];
    while end > 0u && count > 0u {
        alt rposition_between(v, 0u, end, f) {
          none { break }
          some(i) {
            push(result, slice(v, i + 1u, end));
            // Make sure to skip the separator.
            end = i;
            count -= 1u;
          }
        }
    }
    push(result, slice(v, 0u, end));
    reversed(result)
}

// Mutators

/// Removes the first element from a vector and return it
fn shift<T>(&v: ~[T]) -> T {
    let ln = len::<T>(v);
    assert (ln > 0);

    let mut vv = ~[];
    v <-> vv;

    unsafe {
        let mut rr;
        {
            let vv = unsafe::to_ptr(vv);
            rr <- *vv;

            for uint::range(1, ln) |i| {
                let r <- *ptr::offset(vv, i);
                push(v, r);
            }
        }
        unsafe::set_len(vv, 0);

        rr
    }
}

/// Prepend an element to the vector
fn unshift<T>(&v: ~[T], +x: T) {
    let mut vv = ~[x];
    v <-> vv;
    while len(vv) > 0 {
        push(v, shift(vv));
    }
}

fn consume<T>(+v: ~[T], f: fn(uint, +T)) unsafe {
    do unpack_slice(v) |p, ln| {
        for uint::range(0, ln) |i| {
            let x <- *ptr::offset(p, i);
            f(i, x);
        }
    }

    unsafe::set_len(v, 0);
}

/// Remove the last element from a vector and return it
fn pop<T>(&v: ~[const T]) -> T {
    let ln = len(v);
    assert ln > 0u;
    let valptr = ptr::mut_addr_of(v[ln - 1u]);
    unsafe {
        let val <- *valptr;
        unsafe::set_len(v, ln - 1u);
        val
    }
}

/// Append an element to a vector
#[inline(always)]
fn push<T>(&v: ~[const T], +initval: T) {
    unsafe {
        let repr: **unsafe::vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        let fill = (**repr).fill;
        if (**repr).alloc > fill {
            let sz = sys::size_of::<T>();
            (**repr).fill += sz;
            let p = ptr::addr_of((**repr).data);
            let p = ptr::offset(p, fill) as *mut T;
            rusti::move_val_init(*p, initval);
        }
        else {
            push_slow(v, initval);
        }
    }
}

fn push_slow<T>(&v: ~[const T], +initval: T) {
    unsafe {
        let ln = v.len();
        reserve_at_least(v, ln + 1u);
        let repr: **unsafe::vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        let fill = (**repr).fill;
        let sz = sys::size_of::<T>();
        (**repr).fill += sz;
        let p = ptr::addr_of((**repr).data);
        let p = ptr::offset(p, fill) as *mut T;
        rusti::move_val_init(*p, initval);
    }
}

// Unchecked vector indexing
#[inline(always)]
unsafe fn ref<T: copy>(v: &[const T], i: uint) -> T {
    unpack_slice(v, |p, _len| *ptr::offset(p, i))
}

#[inline(always)]
fn push_all<T: copy>(&v: ~[const T], rhs: &[const T]) {
    reserve(v, v.len() + rhs.len());

    for uint::range(0u, rhs.len()) |i| {
        push(v, unsafe { ref(rhs, i) })
    }
}

#[inline(always)]
fn push_all_move<T>(&v: ~[const T], -rhs: ~[const T]) {
    reserve(v, v.len() + rhs.len());
    unsafe {
        do unpack_slice(rhs) |p, len| {
            for uint::range(0, len) |i| {
                let x <- *ptr::offset(p, i);
                push(v, x);
            }
        }
        unsafe::set_len(rhs, 0);
    }
}

// Appending
#[inline(always)]
pure fn append<T: copy>(+lhs: ~[T], rhs: &[const T]) -> ~[T] {
    let mut v <- lhs;
    unchecked {
        push_all(v, rhs);
    }
    ret v;
}

#[inline(always)]
pure fn append_one<T>(+lhs: ~[T], +x: T) -> ~[T] {
    let mut v <- lhs;
    unchecked { push(v, x); }
    v
}

#[inline(always)]
pure fn append_mut<T: copy>(lhs: &[mut T], rhs: &[const T]) -> ~[mut T] {
    let mut v = ~[mut];
    let mut i = 0u;
    while i < lhs.len() {
        unsafe { // This is impure, but it appears pure to the caller.
            push(v, lhs[i]);
        }
        i += 1u;
    }
    i = 0u;
    while i < rhs.len() {
        unsafe { // This is impure, but it appears pure to the caller.
            push(v, rhs[i]);
        }
        i += 1u;
    }
    ret v;
}

/**
 * Expands a vector in place, initializing the new elements to a given value
 *
 * # Arguments
 *
 * * v - The vector to grow
 * * n - The number of elements to add
 * * initval - The value for the new elements
 */
fn grow<T: copy>(&v: ~[const T], n: uint, initval: T) {
    reserve_at_least(v, len(v) + n);
    let mut i: uint = 0u;

    while i < n { push(v, initval); i += 1u; }
}

/**
 * Expands a vector in place, initializing the new elements to the result of
 * a function
 *
 * Function `init_op` is called `n` times with the values [0..`n`)
 *
 * # Arguments
 *
 * * v - The vector to grow
 * * n - The number of elements to add
 * * init_op - A function to call to retreive each appended element's
 *             value
 */
fn grow_fn<T>(&v: ~[const T], n: uint, op: init_op<T>) {
    reserve_at_least(v, len(v) + n);
    let mut i: uint = 0u;
    while i < n { push(v, op(i)); i += 1u; }
}

/**
 * Sets the value of a vector element at a given index, growing the vector as
 * needed
 *
 * Sets the element at position `index` to `val`. If `index` is past the end
 * of the vector, expands the vector by replicating `initval` to fill the
 * intervening space.
 */
#[inline(always)]
fn grow_set<T: copy>(&v: ~[mut T], index: uint, initval: T, val: T) {
    if index >= len(v) { grow(v, index - len(v) + 1u, initval); }
    v[index] = val;
}


// Functional utilities

/// Apply a function to each element of a vector and return the results
pure fn map<T, U>(v: &[T], f: fn(T) -> U) -> ~[U] {
    let mut result = ~[];
    unchecked{reserve(result, len(v));}
    for each(v) |elem| { unsafe { push(result, f(elem)); } }
    ret result;
}

fn map_consume<T, U>(+v: ~[T], f: fn(+T) -> U) -> ~[U] {
    let mut result = ~[];
    do consume(v) |_i, x| {
        vec::push(result, f(x));
    }
    result
}

/// Apply a function to each element of a vector and return the results
pure fn mapi<T, U>(v: &[T], f: fn(uint, T) -> U) -> ~[U] {
    let mut result = ~[];
    unchecked{reserve(result, len(v));}
    for eachi(v) |i, elem| { unsafe { push(result, f(i, elem)); } }
    ret result;
}

/**
 * Apply a function to each element of a vector and return a concatenation
 * of each result vector
 */
pure fn flat_map<T, U>(v: &[T], f: fn(T) -> ~[U]) -> ~[U] {
    let mut result = ~[];
    for each(v) |elem| { unchecked{ push_all_move(result, f(elem)); } }
    ret result;
}

/// Apply a function to each pair of elements and return the results
pure fn map2<T: copy, U: copy, V>(v0: &[T], v1: &[U],
                                  f: fn(T, U) -> V) -> ~[V] {
    let v0_len = len(v0);
    if v0_len != len(v1) { fail; }
    let mut u: ~[V] = ~[];
    let mut i = 0u;
    while i < v0_len {
        unsafe { push(u, f(copy v0[i], copy v1[i])) };
        i += 1u;
    }
    ret u;
}

/**
 * Apply a function to each element of a vector and return the results
 *
 * If function `f` returns `none` then that element is excluded from
 * the resulting vector.
 */
pure fn filter_map<T, U: copy>(v: &[T], f: fn(T) -> option<U>)
    -> ~[U] {
    let mut result = ~[];
    for each(v) |elem| {
        alt f(elem) {
          none {/* no-op */ }
          some(result_elem) { unsafe { push(result, result_elem); } }
        }
    }
    ret result;
}

/**
 * Construct a new vector from the elements of a vector for which some
 * predicate holds.
 *
 * Apply function `f` to each element of `v` and return a vector containing
 * only those elements for which `f` returned true.
 */
pure fn filter<T: copy>(v: &[T], f: fn(T) -> bool) -> ~[T] {
    let mut result = ~[];
    for each(v) |elem| {
        if f(elem) { unsafe { push(result, elem); } }
    }
    ret result;
}

/**
 * Concatenate a vector of vectors.
 *
 * Flattens a vector of vectors of T into a single vector of T.
 */
pure fn concat<T: copy>(v: &[~[T]]) -> ~[T] {
    let mut r = ~[];
    for each(v) |inner| { unsafe { push_all(r, inner); } }
    ret r;
}

/// Concatenate a vector of vectors, placing a given separator between each
pure fn connect<T: copy>(v: &[~[T]], sep: T) -> ~[T] {
    let mut r: ~[T] = ~[];
    let mut first = true;
    for each(v) |inner| {
        if first { first = false; } else { unsafe { push(r, sep); } }
        unchecked { push_all(r, inner) };
    }
    ret r;
}

/// Reduce a vector from left to right
pure fn foldl<T: copy, U>(z: T, v: &[U], p: fn(T, U) -> T) -> T {
    let mut accum = z;
    do iter(v) |elt| {
        accum = p(accum, elt);
    }
    ret accum;
}

/// Reduce a vector from right to left
pure fn foldr<T, U: copy>(v: &[T], z: U, p: fn(T, U) -> U) -> U {
    let mut accum = z;
    do riter(v) |elt| {
        accum = p(elt, accum);
    }
    ret accum;
}

/**
 * Return true if a predicate matches any elements
 *
 * If the vector contains no elements then false is returned.
 */
pure fn any<T>(v: &[T], f: fn(T) -> bool) -> bool {
    for each(v) |elem| { if f(elem) { ret true; } }
    ret false;
}

/**
 * Return true if a predicate matches any elements in both vectors.
 *
 * If the vectors contains no elements then false is returned.
 */
pure fn any2<T, U>(v0: &[T], v1: &[U],
                   f: fn(T, U) -> bool) -> bool {
    let v0_len = len(v0);
    let v1_len = len(v1);
    let mut i = 0u;
    while i < v0_len && i < v1_len {
        if f(v0[i], v1[i]) { ret true; };
        i += 1u;
    }
    ret false;
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pure fn all<T>(v: &[T], f: fn(T) -> bool) -> bool {
    for each(v) |elem| { if !f(elem) { ret false; } }
    ret true;
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pure fn alli<T>(v: &[T], f: fn(uint, T) -> bool) -> bool {
    for eachi(v) |i, elem| { if !f(i, elem) { ret false; } }
    ret true;
}

/**
 * Return true if a predicate matches all elements in both vectors.
 *
 * If the vectors are not the same size then false is returned.
 */
pure fn all2<T, U>(v0: &[T], v1: &[U],
                   f: fn(T, U) -> bool) -> bool {
    let v0_len = len(v0);
    if v0_len != len(v1) { ret false; }
    let mut i = 0u;
    while i < v0_len { if !f(v0[i], v1[i]) { ret false; }; i += 1u; }
    ret true;
}

/// Return true if a vector contains an element with the given value
pure fn contains<T>(v: &[T], x: T) -> bool {
    for each(v) |elt| { if x == elt { ret true; } }
    ret false;
}

/// Returns the number of elements that are equal to a given value
pure fn count<T>(v: &[T], x: T) -> uint {
    let mut cnt = 0u;
    for each(v) |elt| { if x == elt { cnt += 1u; } }
    ret cnt;
}

/**
 * Search for the first element that matches a given predicate
 *
 * Apply function `f` to each element of `v`, starting from the first.
 * When function `f` returns true then an option containing the element
 * is returned. If `f` matches no elements then none is returned.
 */
pure fn find<T: copy>(v: &[T], f: fn(T) -> bool) -> option<T> {
    find_between(v, 0u, len(v), f)
}

/**
 * Search for the first element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is returned.
 */
pure fn find_between<T: copy>(v: &[T], start: uint, end: uint,
                      f: fn(T) -> bool) -> option<T> {
    option::map(position_between(v, start, end, f), |i| v[i])
}

/**
 * Search for the last element that matches a given predicate
 *
 * Apply function `f` to each element of `v` in reverse order. When function
 * `f` returns true then an option containing the element is returned. If `f`
 * matches no elements then none is returned.
 */
pure fn rfind<T: copy>(v: &[T], f: fn(T) -> bool) -> option<T> {
    rfind_between(v, 0u, len(v), f)
}

/**
 * Search for the last element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is returned.
 */
pure fn rfind_between<T: copy>(v: &[T], start: uint, end: uint,
                               f: fn(T) -> bool) -> option<T> {
    option::map(rposition_between(v, start, end, f), |i| v[i])
}

/// Find the first index containing a matching value
pure fn position_elem<T>(v: &[T], x: T) -> option<uint> {
    position(v, |y| x == y)
}

/**
 * Find the first index matching some predicate
 *
 * Apply function `f` to each element of `v`.  When function `f` returns true
 * then an option containing the index is returned. If `f` matches no elements
 * then none is returned.
 */
pure fn position<T>(v: &[T], f: fn(T) -> bool) -> option<uint> {
    position_between(v, 0u, len(v), f)
}

/**
 * Find the first index matching some predicate within a range
 *
 * Apply function `f` to each element of `v` between the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the index is returned. If `f` matches no elements then none is returned.
 */
pure fn position_between<T>(v: &[T], start: uint, end: uint,
                            f: fn(T) -> bool) -> option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = start;
    while i < end { if f(v[i]) { ret some::<uint>(i); } i += 1u; }
    ret none;
}

/// Find the last index containing a matching value
pure fn rposition_elem<T>(v: &[T], x: T) -> option<uint> {
    rposition(v, |y| x == y)
}

/**
 * Find the last index matching some predicate
 *
 * Apply function `f` to each element of `v` in reverse order.  When function
 * `f` returns true then an option containing the index is returned. If `f`
 * matches no elements then none is returned.
 */
pure fn rposition<T>(v: &[T], f: fn(T) -> bool) -> option<uint> {
    rposition_between(v, 0u, len(v), f)
}

/**
 * Find the last index matching some predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order between the
 * range [`start`, `end`). When function `f` returns true then an option
 * containing the index is returned. If `f` matches no elements then none is
 * returned.
 */
pure fn rposition_between<T>(v: &[T], start: uint, end: uint,
                             f: fn(T) -> bool) -> option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = end;
    while i > start {
        if f(v[i - 1u]) { ret some::<uint>(i - 1u); }
        i -= 1u;
    }
    ret none;
}

// FIXME: if issue #586 gets implemented, could have a postcondition
// saying the two result lists have the same length -- or, could
// return a nominal record with a constraint saying that, instead of
// returning a tuple (contingent on issue #869)
/**
 * Convert a vector of pairs into a pair of vectors
 *
 * Returns a tuple containing two vectors where the i-th element of the first
 * vector contains the first element of the i-th tuple of the input vector,
 * and the i-th element of the second vector contains the second element
 * of the i-th tuple of the input vector.
 */
pure fn unzip<T: copy, U: copy>(v: &[(T, U)]) -> (~[T], ~[U]) {
    let mut as = ~[], bs = ~[];
    for each(v) |p| {
        let (a, b) = p;
        unchecked {
            vec::push(as, a);
            vec::push(bs, b);
        }
    }
    ret (as, bs);
}

/**
 * Convert two vectors to a vector of pairs
 *
 * Returns a vector of tuples, where the i-th tuple contains contains the
 * i-th elements from each of the input vectors.
 */
pure fn zip<T: copy, U: copy>(v: &[const T], u: &[const U]) -> ~[(T, U)] {
    let mut zipped = ~[];
    let sz = len(v);
    let mut i = 0u;
    assert sz == len(u);
    while i < sz unchecked { vec::push(zipped, (v[i], u[i])); i += 1u; }
    ret zipped;
}

/**
 * Swaps two elements in a vector
 *
 * # Arguments
 *
 * * v  The input vector
 * * a - The index of the first element
 * * b - The index of the second element
 */
fn swap<T>(&&v: ~[mut T], a: uint, b: uint) {
    v[a] <-> v[b];
}

/// Reverse the order of elements in a vector, in place
fn reverse<T>(v: ~[mut T]) {
    let mut i: uint = 0u;
    let ln = len::<T>(v);
    while i < ln / 2u { v[i] <-> v[ln - i - 1u]; i += 1u; }
}


/// Returns a vector with the order of elements reversed
pure fn reversed<T: copy>(v: &[const T]) -> ~[T] {
    let mut rs: ~[T] = ~[];
    let mut i = len::<T>(v);
    if i == 0u { ret rs; } else { i -= 1u; }
    unchecked {
        while i != 0u { vec::push(rs, v[i]); i -= 1u; }
        vec::push(rs, v[0]);
    }
    ret rs;
}

/**
 * Iterates over a slice
 *
 * Iterates over slice `v` and, for each element, calls function `f` with the
 * element's value.
 */
#[inline(always)]
pure fn iter<T>(v: &[T], f: fn(T)) {
    iter_between(v, 0u, vec::len(v), f)
}

/*
Function: iter_between

Iterates over a slice

Iterates over slice `v` and, for each element, calls function `f` with the
element's value.

*/
#[inline(always)]
pure fn iter_between<T>(v: &[T], start: uint, end: uint, f: fn(T)) {
    do unpack_slice(v) |base_ptr, len| {
        assert start <= end;
        assert end <= len;
        unsafe {
            let mut n = end;
            let mut p = ptr::offset(base_ptr, start);
            while n > start {
                f(*p);
                p = ptr::offset(p, 1u);
                n -= 1u;
            }
        }
    }
}

/**
 * Iterates over a vector, with option to break
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pure fn each<T>(v: &[const T], f: fn(T) -> bool) {
    do vec::unpack_slice(v) |p, n| {
        let mut n = n;
        let mut p = p;
        while n > 0u {
            unsafe {
                if !f(*p) { break; }
                p = ptr::offset(p, 1u);
            }
            n -= 1u;
        }
    }
}

/**
 * Iterates over a vector's elements and indices
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pure fn eachi<T>(v: &[const T], f: fn(uint, T) -> bool) {
    do vec::unpack_slice(v) |p, n| {
        let mut i = 0u;
        let mut p = p;
        while i < n {
            unsafe {
                if !f(i, *p) { break; }
                p = ptr::offset(p, 1u);
            }
            i += 1u;
        }
    }
}

/**
 * Iterates over two vectors simultaneously
 *
 * # Failure
 *
 * Both vectors must have the same length
 */
#[inline]
fn iter2<U, T>(v1: &[U], v2: &[T], f: fn(U, T)) {
    assert len(v1) == len(v2);
    for uint::range(0u, len(v1)) |i| {
        f(v1[i], v2[i])
    }
}

/**
 * Iterates over a vector's elements and indexes
 *
 * Iterates over vector `v` and, for each element, calls function `f` with the
 * element's value and index.
 */
#[inline(always)]
pure fn iteri<T>(v: &[T], f: fn(uint, T)) {
    let mut i = 0u;
    let l = len(v);
    while i < l { f(i, v[i]); i += 1u; }
}

/**
 * Iterates over a vector in reverse
 *
 * Iterates over vector `v` and, for each element, calls function `f` with the
 * element's value.
 */
pure fn riter<T>(v: &[T], f: fn(T)) {
    riteri(v, |_i, v| f(v))
}

/**
 * Iterates over a vector's elements and indexes in reverse
 *
 * Iterates over vector `v` and, for each element, calls function `f` with the
 * element's value and index.
 */
pure fn riteri<T>(v: &[T], f: fn(uint, T)) {
    let mut i = len(v);
    while 0u < i {
        i -= 1u;
        f(i, v[i]);
    };
}

/**
 * Iterate over all permutations of vector `v`.
 *
 * Permutations are produced in lexicographic order with respect to the order
 * of elements in `v` (so if `v` is sorted then the permutations are
 * lexicographically sorted).
 *
 * The total number of permutations produced is `len(v)!`.  If `v` contains
 * repeated elements, then some permutations are repeated.
 */
pure fn permute<T: copy>(v: &[T], put: fn(~[T])) {
    let ln = len(v);
    if ln == 0u {
        put(~[]);
    } else {
        let mut i = 0u;
        while i < ln {
            let elt = v[i];
            let mut rest = slice(v, 0u, i);
            unchecked {
                push_all(rest, view(v, i+1u, ln));
                permute(rest, |permutation| {
                    put(append(~[elt], permutation))
                })
            }
            i += 1u;
        }
    }
}

pure fn windowed<TT: copy>(nn: uint, xx: &[TT]) -> ~[~[TT]] {
    let mut ww = ~[];
    assert 1u <= nn;
    vec::iteri (xx, |ii, _x| {
        let len = vec::len(xx);
        if ii+nn <= len unchecked {
            vec::push(ww, vec::slice(xx, ii, ii+nn));
        }
    });
    ret ww;
}

/**
 * Work with the buffer of a vector.
 *
 * Allows for unsafe manipulation of vector contents, which is useful for
 * foreign interop.
 */
fn as_buf<E,T>(v: &[E], f: fn(*E) -> T) -> T {
    unpack_slice(v, |buf, _len| f(buf))
}

fn as_mut_buf<E,T>(v: &[mut E], f: fn(*mut E) -> T) -> T {
    unpack_mut_slice(v, |buf, _len| f(buf))
}

/// Work with the buffer and length of a slice.
#[inline(always)]
pure fn unpack_slice<T,U>(s: &[const T],
                          f: fn(*T, uint) -> U) -> U {
    unsafe {
        let v : *(*T,uint) = ::unsafe::reinterpret_cast(ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len / sys::size_of::<T>())
    }
}

/// Work with the buffer and length of a slice.
#[inline(always)]
pure fn unpack_const_slice<T,U>(s: &[const T],
                                f: fn(*const T, uint) -> U) -> U {
    unsafe {
        let v : *(*const T,uint) =
            ::unsafe::reinterpret_cast(ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len / sys::size_of::<T>())
    }
}

/// Work with the buffer and length of a slice.
#[inline(always)]
pure fn unpack_mut_slice<T,U>(s: &[mut T],
                              f: fn(*mut T, uint) -> U) -> U {
    unsafe {
        let v : *(*const T,uint) =
            ::unsafe::reinterpret_cast(ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len / sys::size_of::<T>())
    }
}

trait vec_concat<T> {
    pure fn +(rhs: &[const T]) -> self;
}

impl extensions<T: copy> of vec_concat<T> for ~[T] {
    #[inline(always)]
    pure fn +(rhs: &[const T]) -> ~[T] {
        append(self, rhs)
    }
}

impl extensions<T: copy> of vec_concat<T> for ~[mut T] {
    #[inline(always)]
    pure fn +(rhs: &[const T]) -> ~[mut T] {
        append_mut(self, rhs)
    }
}

trait const_vector {
    pure fn is_empty() -> bool;
    pure fn is_not_empty() -> bool;
    pure fn len() -> uint;
}

/// Extension methods for vectors
impl extensions/&<T> of const_vector for &[const T] {
    /// Returns true if a vector contains no elements
    #[inline]
    pure fn is_empty() -> bool { is_empty(self) }
    /// Returns true if a vector contains some elements
    #[inline]
    pure fn is_not_empty() -> bool { is_not_empty(self) }
    /// Returns the length of a vector
    #[inline]
    pure fn len() -> uint { len(self) }
}

trait copyable_vector<T> {
    pure fn head() -> T;
    pure fn init() -> ~[T];
    pure fn last() -> T;
    pure fn slice(start: uint, end: uint) -> ~[T];
    pure fn tail() -> ~[T];
}

/// Extension methods for vectors
impl extensions/&<T: copy> of copyable_vector<T> for &[const T] {
    /// Returns the first element of a vector
    #[inline]
    pure fn head() -> T { head(self) }
    /// Returns all but the last elemnt of a vector
    #[inline]
    pure fn init() -> ~[T] { init(self) }
    /// Returns the last element of a `v`, failing if the vector is empty.
    #[inline]
    pure fn last() -> T { last(self) }
    /// Returns a copy of the elements from [`start`..`end`) from `v`.
    #[inline]
    pure fn slice(start: uint, end: uint) -> ~[T] { slice(self, start, end) }
    /// Returns all but the first element of a vector
    #[inline]
    pure fn tail() -> ~[T] { tail(self) }
}

trait immutable_vector/&<T> {
    pure fn foldr<U: copy>(z: U, p: fn(T, U) -> U) -> U;
    pure fn iter(f: fn(T));
    pure fn iteri(f: fn(uint, T));
    pure fn position(f: fn(T) -> bool) -> option<uint>;
    pure fn position_elem(x: T) -> option<uint>;
    pure fn riter(f: fn(T));
    pure fn riteri(f: fn(uint, T));
    pure fn rposition(f: fn(T) -> bool) -> option<uint>;
    pure fn rposition_elem(x: T) -> option<uint>;
    pure fn map<U>(f: fn(T) -> U) -> ~[U];
    pure fn mapi<U>(f: fn(uint, T) -> U) -> ~[U];
    fn map_r<U>(f: fn(x: &self.T) -> U) -> ~[U];
    pure fn alli(f: fn(uint, T) -> bool) -> bool;
    pure fn flat_map<U>(f: fn(T) -> ~[U]) -> ~[U];
    pure fn filter_map<U: copy>(f: fn(T) -> option<U>) -> ~[U];
}

/// Extension methods for vectors
impl extensions/&<T> of immutable_vector<T> for &[T] {
    /// Reduce a vector from right to left
    #[inline]
    pure fn foldr<U: copy>(z: U, p: fn(T, U) -> U) -> U { foldr(self, z, p) }
    /**
     * Iterates over a vector
     *
     * Iterates over vector `v` and, for each element, calls function `f` with
     * the element's value.
     */
    #[inline]
    pure fn iter(f: fn(T)) { iter(self, f) }
    /**
     * Iterates over a vector's elements and indexes
     *
     * Iterates over vector `v` and, for each element, calls function `f` with
     * the element's value and index.
     */
    #[inline]
    pure fn iteri(f: fn(uint, T)) { iteri(self, f) }
    /**
     * Find the first index matching some predicate
     *
     * Apply function `f` to each element of `v`.  When function `f` returns
     * true then an option containing the index is returned. If `f` matches no
     * elements then none is returned.
     */
    #[inline]
    pure fn position(f: fn(T) -> bool) -> option<uint> { position(self, f) }
    /// Find the first index containing a matching value
    #[inline]
    pure fn position_elem(x: T) -> option<uint> { position_elem(self, x) }
    /**
     * Iterates over a vector in reverse
     *
     * Iterates over vector `v` and, for each element, calls function `f` with
     * the element's value.
     */
    #[inline]
    pure fn riter(f: fn(T)) { riter(self, f) }
    /**
     * Iterates over a vector's elements and indexes in reverse
     *
     * Iterates over vector `v` and, for each element, calls function `f` with
     * the element's value and index.
     */
    #[inline]
    pure fn riteri(f: fn(uint, T)) { riteri(self, f) }
    /**
     * Find the last index matching some predicate
     *
     * Apply function `f` to each element of `v` in reverse order.  When
     * function `f` returns true then an option containing the index is
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    pure fn rposition(f: fn(T) -> bool) -> option<uint> { rposition(self, f) }
    /// Find the last index containing a matching value
    #[inline]
    pure fn rposition_elem(x: T) -> option<uint> { rposition_elem(self, x) }
    /// Apply a function to each element of a vector and return the results
    #[inline]
    pure fn map<U>(f: fn(T) -> U) -> ~[U] { map(self, f) }
    /**
     * Apply a function to the index and value of each element in the vector
     * and return the results
     */
    pure fn mapi<U>(f: fn(uint, T) -> U) -> ~[U] {
        mapi(self, f)
    }

    #[inline]
    fn map_r<U>(f: fn(x: &self.T) -> U) -> ~[U] {
        let mut r = ~[];
        let mut i = 0;
        while i < self.len() {
            push(r, f(&self[i]));
            i += 1;
        }
        r
    }

    /**
     * Returns true if the function returns true for all elements.
     *
     *     If the vector is empty, true is returned.
     */
    pure fn alli(f: fn(uint, T) -> bool) -> bool {
        alli(self, f)
    }
    /**
     * Apply a function to each element of a vector and return a concatenation
     * of each result vector
     */
    #[inline]
    pure fn flat_map<U>(f: fn(T) -> ~[U]) -> ~[U] { flat_map(self, f) }
    /**
     * Apply a function to each element of a vector and return the results
     *
     * If function `f` returns `none` then that element is excluded from
     * the resulting vector.
     */
    #[inline]
    pure fn filter_map<U: copy>(f: fn(T) -> option<U>) -> ~[U] {
        filter_map(self, f)
    }
}

trait immutable_copyable_vector<T> {
    pure fn filter(f: fn(T) -> bool) -> ~[T];
    pure fn find(f: fn(T) -> bool) -> option<T>;
    pure fn rfind(f: fn(T) -> bool) -> option<T>;
}

/// Extension methods for vectors
impl extensions/&<T: copy> of immutable_copyable_vector<T> for &[T] {
    /**
     * Construct a new vector from the elements of a vector for which some
     * predicate holds.
     *
     * Apply function `f` to each element of `v` and return a vector
     * containing only those elements for which `f` returned true.
     */
    #[inline]
    pure fn filter(f: fn(T) -> bool) -> ~[T] { filter(self, f) }
    /**
     * Search for the first element that matches a given predicate
     *
     * Apply function `f` to each element of `v`, starting from the first.
     * When function `f` returns true then an option containing the element
     * is returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    pure fn find(f: fn(T) -> bool) -> option<T> { find(self, f) }
    /**
     * Search for the last element that matches a given predicate
     *
     * Apply function `f` to each element of `v` in reverse order. When
     * function `f` returns true then an option containing the element is
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    pure fn rfind(f: fn(T) -> bool) -> option<T> { rfind(self, f) }
}

/// Unsafe operations
mod unsafe {
    // FIXME: This should have crate visibility (#1893 blocks that)
    /// The internal representation of a vector
    type vec_repr = {
        box_header: (uint, uint, uint, uint),
        mut fill: uint,
        mut alloc: uint,
        data: u8
    };

    type slice_repr = {
        mut data: *u8,
        mut len: uint
    };

    /**
     * Constructs a vector from an unsafe pointer to a buffer
     *
     * # Arguments
     *
     * * ptr - An unsafe pointer to a buffer of `T`
     * * elts - The number of elements in the buffer
     */
    #[inline(always)]
    unsafe fn from_buf<T>(ptr: *T, elts: uint) -> ~[T] {
        ret ::unsafe::reinterpret_cast(
            rustrt::vec_from_buf_shared(sys::get_type_desc::<T>(),
                                        ptr as *(),
                                        elts as size_t));
    }

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifing its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline(always)]
    unsafe fn set_len<T>(&&v: ~[const T], new_len: uint) {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).fill = new_len * sys::size_of::<T>();
    }

    /**
     * Returns an unsafe pointer to the vector's buffer
     *
     * The caller must ensure that the vector outlives the pointer this
     * function returns, or else it will end up pointing to garbage.
     *
     * Modifying the vector may cause its buffer to be reallocated, which
     * would also make any pointers to it invalid.
     */
    #[inline(always)]
    unsafe fn to_ptr<T>(v: ~[const T]) -> *T {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        ret ::unsafe::reinterpret_cast(addr_of((**repr).data));
    }


    #[inline(always)]
    unsafe fn to_ptr_slice<T>(v: &[const T]) -> *T {
        let repr: **slice_repr = ::unsafe::reinterpret_cast(addr_of(v));
        ret ::unsafe::reinterpret_cast(addr_of((**repr).data));
    }


    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline(always)]
    unsafe fn form_slice<T,U>(p: *T, len: uint, f: fn(&& &[T]) -> U) -> U {
        let pair = (p, len * sys::size_of::<T>());
        let v : *(&blk.[T]) =
            ::unsafe::reinterpret_cast(ptr::addr_of(pair));
        f(*v)
    }
}

/// Operations on `[u8]`
mod u8 {
    export cmp;
    export lt, le, eq, ne, ge, gt;
    export hash;

    /// Bytewise string comparison
    pure fn cmp(&&a: ~[u8], &&b: ~[u8]) -> int {
        let a_len = len(a);
        let b_len = len(b);
        let n = uint::min(a_len, b_len) as libc::size_t;
        let r = unsafe {
            libc::memcmp(unsafe::to_ptr(a) as *libc::c_void,
                         unsafe::to_ptr(b) as *libc::c_void, n) as int
        };

        if r != 0 { r } else {
            if a_len == b_len {
                0
            } else if a_len < b_len {
                -1
            } else {
                1
            }
        }
    }

    /// Bytewise less than or equal
    pure fn lt(&&a: ~[u8], &&b: ~[u8]) -> bool { cmp(a, b) < 0 }

    /// Bytewise less than or equal
    pure fn le(&&a: ~[u8], &&b: ~[u8]) -> bool { cmp(a, b) <= 0 }

    /// Bytewise equality
    pure fn eq(&&a: ~[u8], &&b: ~[u8]) -> bool { unsafe { cmp(a, b) == 0 } }

    /// Bytewise inequality
    pure fn ne(&&a: ~[u8], &&b: ~[u8]) -> bool { unsafe { cmp(a, b) != 0 } }

    /// Bytewise greater than or equal
    pure fn ge(&&a: ~[u8], &&b: ~[u8]) -> bool { cmp(a, b) >= 0 }

    /// Bytewise greater than
    pure fn gt(&&a: ~[u8], &&b: ~[u8]) -> bool { cmp(a, b) > 0 }

    /// String hash function
    fn hash(&&s: ~[u8]) -> uint {
        /* Seems to have been tragically copy/pasted from str.rs,
           or vice versa. But I couldn't figure out how to abstract
           it out. -- tjc */

        let mut u: uint = 5381u;
        vec::iter(s, |c| {u *= 33u; u += c as uint;});
        ret u;
    }
}

// ___________________________________________________________________________
// ITERATION TRAIT METHODS
//
// This cannot be used with iter-trait.rs because of the region pointer
// required in the slice.
impl extensions/&<A> of iter::base_iter<A> for &[const A] {
    fn each(blk: fn(A) -> bool) { each(self, blk) }
    fn size_hint() -> option<uint> { some(len(self)) }
    fn eachi(blk: fn(uint, A) -> bool) { iter::eachi(self, blk) }
    fn all(blk: fn(A) -> bool) -> bool { iter::all(self, blk) }
    fn any(blk: fn(A) -> bool) -> bool { iter::any(self, blk) }
    fn foldl<B>(+b0: B, blk: fn(B, A) -> B) -> B {
        iter::foldl(self, b0, blk)
    }
    fn contains(x: A) -> bool { iter::contains(self, x) }
    fn count(x: A) -> uint { iter::count(self, x) }
}

trait iter_trait_extensions<A> {
    fn filter_to_vec(pred: fn(A) -> bool) -> ~[A];
    fn map_to_vec<B>(op: fn(A) -> B) -> ~[B];
    fn to_vec() -> ~[A];
    fn min() -> A;
    fn max() -> A;
}

impl extensions/&<A:copy> of iter_trait_extensions<A> for &[const A] {
    fn filter_to_vec(pred: fn(A) -> bool) -> ~[A] {
        iter::filter_to_vec(self, pred)
    }
    fn map_to_vec<B>(op: fn(A) -> B) -> ~[B] { iter::map_to_vec(self, op) }
    fn to_vec() -> ~[A] { iter::to_vec(self) }

    // FIXME--bug in resolve prevents this from working (#2611)
    // fn flat_map_to_vec<B:copy,IB:base_iter<B>>(op: fn(A) -> IB) -> ~[B] {
    //     iter::flat_map_to_vec(self, op)
    // }

    fn min() -> A { iter::min(self) }
    fn max() -> A { iter::max(self) }
}
// ___________________________________________________________________________

#[cfg(test)]
mod tests {

    fn square(n: uint) -> uint { ret n * n; }

    fn square_ref(&&n: uint) -> uint { ret n * n; }

    pure fn is_three(&&n: uint) -> bool { ret n == 3u; }

    pure fn is_odd(&&n: uint) -> bool { ret n % 2u == 1u; }

    pure fn is_equal(&&x: uint, &&y:uint) -> bool { ret x == y; }

    fn square_if_odd(&&n: uint) -> option<uint> {
        ret if n % 2u == 1u { some(n * n) } else { none };
    }

    fn add(&&x: uint, &&y: uint) -> uint { ret x + y; }

    #[test]
    fn test_unsafe_ptrs() {
        unsafe {
            // Test on-stack copy-from-buf.
            let a = ~[1, 2, 3];
            let mut ptr = unsafe::to_ptr(a);
            let b = unsafe::from_buf(ptr, 3u);
            assert (len(b) == 3u);
            assert (b[0] == 1);
            assert (b[1] == 2);
            assert (b[2] == 3);

            // Test on-heap copy-from-buf.
            let c = ~[1, 2, 3, 4, 5];
            ptr = unsafe::to_ptr(c);
            let d = unsafe::from_buf(ptr, 5u);
            assert (len(d) == 5u);
            assert (d[0] == 1);
            assert (d[1] == 2);
            assert (d[2] == 3);
            assert (d[3] == 4);
            assert (d[4] == 5);
        }
    }

    #[test]
    fn test_from_fn() {
        // Test on-stack from_fn.
        let mut v = from_fn(3u, square);
        assert (len(v) == 3u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);

        // Test on-heap from_fn.
        v = from_fn(5u, square);
        assert (len(v) == 5u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);
        assert (v[3] == 9u);
        assert (v[4] == 16u);
    }

    #[test]
    fn test_from_elem() {
        // Test on-stack from_elem.
        let mut v = from_elem(2u, 10u);
        assert (len(v) == 2u);
        assert (v[0] == 10u);
        assert (v[1] == 10u);

        // Test on-heap from_elem.
        v = from_elem(6u, 20u);
        assert (v[0] == 20u);
        assert (v[1] == 20u);
        assert (v[2] == 20u);
        assert (v[3] == 20u);
        assert (v[4] == 20u);
        assert (v[5] == 20u);
    }

    #[test]
    fn test_is_empty() {
        assert (is_empty::<int>(~[]));
        assert (!is_empty(~[0]));
    }

    #[test]
    fn test_is_not_empty() {
        assert (is_not_empty(~[0]));
        assert (!is_not_empty::<int>(~[]));
    }

    #[test]
    fn test_head() {
        let a = ~[11, 12];
        assert (head(a) == 11);
    }

    #[test]
    fn test_tail() {
        let mut a = ~[11];
        assert (tail(a) == ~[]);

        a = ~[11, 12];
        assert (tail(a) == ~[12]);
    }

    #[test]
    fn test_last() {
        let mut n = last_opt(~[]);
        assert (n == none);
        n = last_opt(~[1, 2, 3]);
        assert (n == some(3));
        n = last_opt(~[1, 2, 3, 4, 5]);
        assert (n == some(5));
    }

    #[test]
    fn test_slice() {
        // Test on-stack -> on-stack slice.
        let mut v = slice(~[1, 2, 3], 1u, 3u);
        assert (len(v) == 2u);
        assert (v[0] == 2);
        assert (v[1] == 3);

        // Test on-heap -> on-stack slice.
        v = slice(~[1, 2, 3, 4, 5], 0u, 3u);
        assert (len(v) == 3u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);

        // Test on-heap -> on-heap slice.
        v = slice(~[1, 2, 3, 4, 5, 6], 1u, 6u);
        assert (len(v) == 5u);
        assert (v[0] == 2);
        assert (v[1] == 3);
        assert (v[2] == 4);
        assert (v[3] == 5);
        assert (v[4] == 6);
    }

    #[test]
    fn test_pop() {
        // Test on-stack pop.
        let mut v = ~[1, 2, 3];
        let mut e = pop(v);
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (e == 3);

        // Test on-heap pop.
        v = ~[1, 2, 3, 4, 5];
        e = pop(v);
        assert (len(v) == 4u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);
        assert (v[3] == 4);
        assert (e == 5);
    }

    #[test]
    fn test_push() {
        // Test on-stack push().
        let mut v = ~[];
        push(v, 1);
        assert (len(v) == 1u);
        assert (v[0] == 1);

        // Test on-heap push().
        push(v, 2);
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 2);
    }

    #[test]
    fn test_grow() {
        // Test on-stack grow().
        let mut v = ~[];
        grow(v, 2u, 1);
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 1);

        // Test on-heap grow().
        grow(v, 3u, 2);
        assert (len(v) == 5u);
        assert (v[0] == 1);
        assert (v[1] == 1);
        assert (v[2] == 2);
        assert (v[3] == 2);
        assert (v[4] == 2);
    }

    #[test]
    fn test_grow_fn() {
        let mut v = ~[];
        grow_fn(v, 3u, square);
        assert (len(v) == 3u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);
    }

    #[test]
    fn test_grow_set() {
        let mut v = ~[mut 1, 2, 3];
        grow_set(v, 4u, 4, 5);
        assert (len(v) == 5u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);
        assert (v[3] == 4);
        assert (v[4] == 5);
    }

    #[test]
    fn test_map() {
        // Test on-stack map.
        let mut v = ~[1u, 2u, 3u];
        let mut w = map(v, square_ref);
        assert (len(w) == 3u);
        assert (w[0] == 1u);
        assert (w[1] == 4u);
        assert (w[2] == 9u);

        // Test on-heap map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = map(v, square_ref);
        assert (len(w) == 5u);
        assert (w[0] == 1u);
        assert (w[1] == 4u);
        assert (w[2] == 9u);
        assert (w[3] == 16u);
        assert (w[4] == 25u);
    }

    #[test]
    fn test_map2() {
        fn times(&&x: int, &&y: int) -> int { ret x * y; }
        let f = times;
        let v0 = ~[1, 2, 3, 4, 5];
        let v1 = ~[5, 4, 3, 2, 1];
        let u = map2::<int, int, int>(v0, v1, f);
        let mut i = 0;
        while i < 5 { assert (v0[i] * v1[i] == u[i]); i += 1; }
    }

    #[test]
    fn test_filter_map() {
        // Test on-stack filter-map.
        let mut v = ~[1u, 2u, 3u];
        let mut w = filter_map(v, square_if_odd);
        assert (len(w) == 2u);
        assert (w[0] == 1u);
        assert (w[1] == 9u);

        // Test on-heap filter-map.
        v = ~[1u, 2u, 3u, 4u, 5u];
        w = filter_map(v, square_if_odd);
        assert (len(w) == 3u);
        assert (w[0] == 1u);
        assert (w[1] == 9u);
        assert (w[2] == 25u);

        fn halve(&&i: int) -> option<int> {
            if i % 2 == 0 {
                ret option::some::<int>(i / 2);
            } else { ret option::none::<int>; }
        }
        fn halve_for_sure(&&i: int) -> int { ret i / 2; }
        let all_even: ~[int] = ~[0, 2, 8, 6];
        let all_odd1: ~[int] = ~[1, 7, 3];
        let all_odd2: ~[int] = ~[];
        let mix: ~[int] = ~[9, 2, 6, 7, 1, 0, 0, 3];
        let mix_dest: ~[int] = ~[1, 3, 0, 0];
        assert (filter_map(all_even, halve) == map(all_even, halve_for_sure));
        assert (filter_map(all_odd1, halve) == ~[]);
        assert (filter_map(all_odd2, halve) == ~[]);
        assert (filter_map(mix, halve) == mix_dest);
    }

    #[test]
    fn test_filter() {
        assert filter(~[1u, 2u, 3u], is_odd) == ~[1u, 3u];
        assert filter(~[1u, 2u, 4u, 8u, 16u], is_three) == ~[];
    }

    #[test]
    fn test_foldl() {
        // Test on-stack fold.
        let mut v = ~[1u, 2u, 3u];
        let mut sum = foldl(0u, v, add);
        assert (sum == 6u);

        // Test on-heap fold.
        v = ~[1u, 2u, 3u, 4u, 5u];
        sum = foldl(0u, v, add);
        assert (sum == 15u);
    }

    #[test]
    fn test_foldl2() {
        fn sub(&&a: int, &&b: int) -> int {
            a - b
        }
        let mut v = ~[1, 2, 3, 4];
        let sum = foldl(0, v, sub);
        assert sum == -10;
    }

    #[test]
    fn test_foldr() {
        fn sub(&&a: int, &&b: int) -> int {
            a - b
        }
        let mut v = ~[1, 2, 3, 4];
        let sum = foldr(v, 0, sub);
        assert sum == -2;
    }

    #[test]
    fn test_iter_empty() {
        let mut i = 0;
        iter::<int>(~[], |_v| i += 1);
        assert i == 0;
    }

    #[test]
    fn test_iter_nonempty() {
        let mut i = 0;
        iter(~[1, 2, 3], |v| i += v);
        assert i == 6;
    }

    #[test]
    fn test_iteri() {
        let mut i = 0;
        iteri(~[1, 2, 3], |j, v| {
            if i == 0 { assert v == 1; }
            assert j + 1u == v as uint;
            i += v;
        });
        assert i == 6;
    }

    #[test]
    fn test_riter_empty() {
        let mut i = 0;
        riter::<int>(~[], |_v| i += 1);
        assert i == 0;
    }

    #[test]
    fn test_riter_nonempty() {
        let mut i = 0;
        riter(~[1, 2, 3], |v| {
            if i == 0 { assert v == 3; }
            i += v
        });
        assert i == 6;
    }

    #[test]
    fn test_riteri() {
        let mut i = 0;
        riteri(~[0, 1, 2], |j, v| {
            if i == 0 { assert v == 2; }
            assert j == v as uint;
            i += v;
        });
        assert i == 3;
    }

    #[test]
    fn test_permute() {
        let mut results: ~[~[int]];

        results = ~[];
        permute(~[], |v| vec::push(results, v));
        assert results == ~[~[]];

        results = ~[];
        permute(~[7], |v| results += ~[v]);
        assert results == ~[~[7]];

        results = ~[];
        permute(~[1,1], |v| results += ~[v]);
        assert results == ~[~[1,1],~[1,1]];

        results = ~[];
        permute(~[5,2,0], |v| results += ~[v]);
        assert results ==
            ~[~[5,2,0],~[5,0,2],~[2,5,0],~[2,0,5],~[0,5,2],~[0,2,5]];
    }

    #[test]
    fn test_any_and_all() {
        assert (any(~[1u, 2u, 3u], is_three));
        assert (!any(~[0u, 1u, 2u], is_three));
        assert (any(~[1u, 2u, 3u, 4u, 5u], is_three));
        assert (!any(~[1u, 2u, 4u, 5u, 6u], is_three));

        assert (all(~[3u, 3u, 3u], is_three));
        assert (!all(~[3u, 3u, 2u], is_three));
        assert (all(~[3u, 3u, 3u, 3u, 3u], is_three));
        assert (!all(~[3u, 3u, 0u, 1u, 2u], is_three));
    }

    #[test]
    fn test_any2_and_all2() {

        assert (any2(~[2u, 4u, 6u], ~[2u, 4u, 6u], is_equal));
        assert (any2(~[1u, 2u, 3u], ~[4u, 5u, 3u], is_equal));
        assert (!any2(~[1u, 2u, 3u], ~[4u, 5u, 6u], is_equal));
        assert (any2(~[2u, 4u, 6u], ~[2u, 4u], is_equal));

        assert (all2(~[2u, 4u, 6u], ~[2u, 4u, 6u], is_equal));
        assert (!all2(~[1u, 2u, 3u], ~[4u, 5u, 3u], is_equal));
        assert (!all2(~[1u, 2u, 3u], ~[4u, 5u, 6u], is_equal));
        assert (!all2(~[2u, 4u, 6u], ~[2u, 4u], is_equal));
    }

    #[test]
    fn test_zip_unzip() {
        let v1 = ~[1, 2, 3];
        let v2 = ~[4, 5, 6];

        let z1 = zip(v1, v2);

        assert ((1, 4) == z1[0]);
        assert ((2, 5) == z1[1]);
        assert ((3, 6) == z1[2]);

        let (left, right) = unzip(z1);

        assert ((1, 4) == (left[0], right[0]));
        assert ((2, 5) == (left[1], right[1]));
        assert ((3, 6) == (left[2], right[2]));
    }

    #[test]
    fn test_position_elem() {
        assert position_elem(~[], 1) == none;

        let v1 = ~[1, 2, 3, 3, 2, 5];
        assert position_elem(v1, 1) == some(0u);
        assert position_elem(v1, 2) == some(1u);
        assert position_elem(v1, 5) == some(5u);
        assert position_elem(v1, 4) == none;
    }

    #[test]
    fn test_position() {
        fn less_than_three(&&i: int) -> bool { ret i < 3; }
        fn is_eighteen(&&i: int) -> bool { ret i == 18; }

        assert position(~[], less_than_three) == none;

        let v1 = ~[5, 4, 3, 2, 1];
        assert position(v1, less_than_three) == some(3u);
        assert position(v1, is_eighteen) == none;
    }

    #[test]
    fn test_position_between() {
        assert position_between(~[], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position_between(v, 0u, 0u, f) == none;
        assert position_between(v, 0u, 1u, f) == none;
        assert position_between(v, 0u, 2u, f) == some(1u);
        assert position_between(v, 0u, 3u, f) == some(1u);
        assert position_between(v, 0u, 4u, f) == some(1u);

        assert position_between(v, 1u, 1u, f) == none;
        assert position_between(v, 1u, 2u, f) == some(1u);
        assert position_between(v, 1u, 3u, f) == some(1u);
        assert position_between(v, 1u, 4u, f) == some(1u);

        assert position_between(v, 2u, 2u, f) == none;
        assert position_between(v, 2u, 3u, f) == none;
        assert position_between(v, 2u, 4u, f) == some(3u);

        assert position_between(v, 3u, 3u, f) == none;
        assert position_between(v, 3u, 4u, f) == some(3u);

        assert position_between(v, 4u, 4u, f) == none;
    }

    #[test]
    fn test_find() {
        assert find(~[], f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find(v, f) == some((1, 'b'));
        assert find(v, g) == none;
    }

    #[test]
    fn test_find_between() {
        assert find_between(~[], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find_between(v, 0u, 0u, f) == none;
        assert find_between(v, 0u, 1u, f) == none;
        assert find_between(v, 0u, 2u, f) == some((1, 'b'));
        assert find_between(v, 0u, 3u, f) == some((1, 'b'));
        assert find_between(v, 0u, 4u, f) == some((1, 'b'));

        assert find_between(v, 1u, 1u, f) == none;
        assert find_between(v, 1u, 2u, f) == some((1, 'b'));
        assert find_between(v, 1u, 3u, f) == some((1, 'b'));
        assert find_between(v, 1u, 4u, f) == some((1, 'b'));

        assert find_between(v, 2u, 2u, f) == none;
        assert find_between(v, 2u, 3u, f) == none;
        assert find_between(v, 2u, 4u, f) == some((3, 'b'));

        assert find_between(v, 3u, 3u, f) == none;
        assert find_between(v, 3u, 4u, f) == some((3, 'b'));

        assert find_between(v, 4u, 4u, f) == none;
    }

    #[test]
    fn test_rposition() {
        assert find(~[], f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position(v, f) == some(1u);
        assert position(v, g) == none;
    }

    #[test]
    fn test_rposition_between() {
        assert rposition_between(~[], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rposition_between(v, 0u, 0u, f) == none;
        assert rposition_between(v, 0u, 1u, f) == none;
        assert rposition_between(v, 0u, 2u, f) == some(1u);
        assert rposition_between(v, 0u, 3u, f) == some(1u);
        assert rposition_between(v, 0u, 4u, f) == some(3u);

        assert rposition_between(v, 1u, 1u, f) == none;
        assert rposition_between(v, 1u, 2u, f) == some(1u);
        assert rposition_between(v, 1u, 3u, f) == some(1u);
        assert rposition_between(v, 1u, 4u, f) == some(3u);

        assert rposition_between(v, 2u, 2u, f) == none;
        assert rposition_between(v, 2u, 3u, f) == none;
        assert rposition_between(v, 2u, 4u, f) == some(3u);

        assert rposition_between(v, 3u, 3u, f) == none;
        assert rposition_between(v, 3u, 4u, f) == some(3u);

        assert rposition_between(v, 4u, 4u, f) == none;
    }

    #[test]
    fn test_rfind() {
        assert rfind(~[], f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind(v, f) == some((3, 'b'));
        assert rfind(v, g) == none;
    }

    #[test]
    fn test_rfind_between() {
        assert rfind_between(~[], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind_between(v, 0u, 0u, f) == none;
        assert rfind_between(v, 0u, 1u, f) == none;
        assert rfind_between(v, 0u, 2u, f) == some((1, 'b'));
        assert rfind_between(v, 0u, 3u, f) == some((1, 'b'));
        assert rfind_between(v, 0u, 4u, f) == some((3, 'b'));

        assert rfind_between(v, 1u, 1u, f) == none;
        assert rfind_between(v, 1u, 2u, f) == some((1, 'b'));
        assert rfind_between(v, 1u, 3u, f) == some((1, 'b'));
        assert rfind_between(v, 1u, 4u, f) == some((3, 'b'));

        assert rfind_between(v, 2u, 2u, f) == none;
        assert rfind_between(v, 2u, 3u, f) == none;
        assert rfind_between(v, 2u, 4u, f) == some((3, 'b'));

        assert rfind_between(v, 3u, 3u, f) == none;
        assert rfind_between(v, 3u, 4u, f) == some((3, 'b'));

        assert rfind_between(v, 4u, 4u, f) == none;
    }

    #[test]
    fn reverse_and_reversed() {
        let v: ~[mut int] = ~[mut 10, 20];
        assert (v[0] == 10);
        assert (v[1] == 20);
        reverse(v);
        assert (v[0] == 20);
        assert (v[1] == 10);
        let v2 = reversed::<int>(~[10, 20]);
        assert (v2[0] == 20);
        assert (v2[1] == 10);
        v[0] = 30;
        assert (v2[0] == 20);
        // Make sure they work with 0-length vectors too.

        let v4 = reversed::<int>(~[]);
        assert (v4 == ~[]);
        let v3: ~[mut int] = ~[mut];
        reverse::<int>(v3);
    }

    #[test]
    fn reversed_mut() {
        let v2 = reversed::<int>(~[mut 10, 20]);
        assert (v2[0] == 20);
        assert (v2[1] == 10);
    }

    #[test]
    fn test_init() {
        let v = init(~[1, 2, 3]);
        assert v == ~[1, 2];
    }

    #[test]
    fn test_split() {
        fn f(&&x: int) -> bool { x == 3 }

        assert split(~[], f) == ~[];
        assert split(~[1, 2], f) == ~[~[1, 2]];
        assert split(~[3, 1, 2], f) == ~[~[], ~[1, 2]];
        assert split(~[1, 2, 3], f) == ~[~[1, 2], ~[]];
        assert split(~[1, 2, 3, 4, 3, 5], f) == ~[~[1, 2], ~[4], ~[5]];
    }

    #[test]
    fn test_splitn() {
        fn f(&&x: int) -> bool { x == 3 }

        assert splitn(~[], 1u, f) == ~[];
        assert splitn(~[1, 2], 1u, f) == ~[~[1, 2]];
        assert splitn(~[3, 1, 2], 1u, f) == ~[~[], ~[1, 2]];
        assert splitn(~[1, 2, 3], 1u, f) == ~[~[1, 2], ~[]];
        assert splitn(~[1, 2, 3, 4, 3, 5], 1u, f) ==
                      ~[~[1, 2], ~[4, 3, 5]];
    }

    #[test]
    fn test_rsplit() {
        fn f(&&x: int) -> bool { x == 3 }

        assert rsplit(~[], f) == ~[];
        assert rsplit(~[1, 2], f) == ~[~[1, 2]];
        assert rsplit(~[1, 2, 3], f) == ~[~[1, 2], ~[]];
        assert rsplit(~[1, 2, 3, 4, 3, 5], f) == ~[~[1, 2], ~[4], ~[5]];
    }

    #[test]
    fn test_rsplitn() {
        fn f(&&x: int) -> bool { x == 3 }

        assert rsplitn(~[], 1u, f) == ~[];
        assert rsplitn(~[1, 2], 1u, f) == ~[~[1, 2]];
        assert rsplitn(~[1, 2, 3], 1u, f) == ~[~[1, 2], ~[]];
        assert rsplitn(~[1, 2, 3, 4, 3, 5], 1u, f) ==
                       ~[~[1, 2, 3, 4], ~[5]];
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_init_empty() {
        init::<int>(~[]);
    }

    #[test]
    fn test_concat() {
        assert concat(~[~[1], ~[2,3]]) == ~[1, 2, 3];
    }

    #[test]
    fn test_connect() {
        assert connect(~[], 0) == ~[];
        assert connect(~[~[1], ~[2, 3]], 0) == ~[1, 0, 2, 3];
        assert connect(~[~[1], ~[2], ~[3]], 0) == ~[1, 0, 2, 0, 3];
    }

    #[test]
    fn test_windowed () {
        assert ~[~[1u,2u,3u],~[2u,3u,4u],~[3u,4u,5u],~[4u,5u,6u]]
            == windowed (3u, ~[1u,2u,3u,4u,5u,6u]);

        assert ~[~[1u,2u,3u,4u],~[2u,3u,4u,5u],~[3u,4u,5u,6u]]
            == windowed (4u, ~[1u,2u,3u,4u,5u,6u]);

        assert ~[] == windowed (7u, ~[1u,2u,3u,4u,5u,6u]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_windowed_() {
        let _x = windowed (0u, ~[1u,2u,3u,4u,5u,6u]);
    }

    #[test]
    fn to_mut_no_copy() {
        unsafe {
            let x = ~[1, 2, 3];
            let addr = unsafe::to_ptr(x);
            let x_mut = to_mut(x);
            let addr_mut = unsafe::to_ptr(x_mut);
            assert addr == addr_mut;
        }
    }

    #[test]
    fn from_mut_no_copy() {
        unsafe {
            let x = ~[mut 1, 2, 3];
            let addr = unsafe::to_ptr(x);
            let x_imm = from_mut(x);
            let addr_imm = unsafe::to_ptr(x_imm);
            assert addr == addr_imm;
        }
    }

    #[test]
    fn test_unshift() {
        let mut x = ~[1, 2, 3];
        unshift(x, 0);
        assert x == ~[0, 1, 2, 3];
    }

    #[test]
    fn test_capacity() {
        let mut v = ~[0u64];
        reserve(v, 10u);
        assert capacity(v) == 10u;
        let mut v = ~[0u32];
        reserve(v, 10u);
        assert capacity(v) == 10u;
    }

    #[test]
    fn test_view() {
        let v = ~[1, 2, 3, 4, 5];
        let v = view(v, 1u, 3u);
        assert(len(v) == 2u);
        assert(v[0] == 2);
        assert(v[1] == 3);
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
