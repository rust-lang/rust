//! Vectors

use cmp::{Eq, Ord};
use option::{Some, None};
use ptr::addr_of;
use libc::size_t;

export append;
export append_one;
export consume, consume_mut;
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
export from_slice;
export with_capacity;
export build, build_sized, build_sized_opt;
export to_mut;
export from_mut;
export head;
export tail;
export tailn;
export init;
export last;
export last_opt;
export slice;
export view, mut_view, const_view;
export split;
export splitn;
export rsplit;
export rsplitn;
export shift;
export unshift;
export pop;
export swap_remove;
export push, push_all, push_all_move;
export grow;
export grow_fn;
export grow_set;
export truncate;
export dedup;
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
export rposition;
export rposition_between;
export unzip;
export zip, zip_slice;
export swap;
export reverse;
export reversed;
export each, each_mut, each_const, eachi, rev_each, rev_eachi;
export iter2;
export permute;
export windowed;
export as_imm_buf;
export as_mut_buf;
export as_const_buf;
export raw;
export bytes;
export extensions;
export ConstVector;
export CopyableVector;
export ImmutableVector;
export ImmutableEqVector;
export ImmutableCopyableVector;
export MutableVector;
export MutableCopyableVector;
export IterTraitExtensions;
export vec_concat;
export traits;

#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports];
    fn vec_reserve_shared(++t: *sys::TypeDesc,
                          ++v: **raw::VecRepr,
                          ++n: libc::size_t);
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn move_val_init<T>(&dst: T, -src: T);
}

/// Returns true if a vector contains no elements
pure fn is_empty<T>(v: &[const T]) -> bool {
    as_const_buf(v, |_p, len| len == 0u)
}

/// Returns true if a vector contains some elements
pure fn is_not_empty<T>(v: &[const T]) -> bool {
    as_const_buf(v, |_p, len| len > 0u)
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
fn reserve<T>(+v: &mut ~[T], +n: uint) {
    // Only make the (slow) call into the runtime if we have to
    if capacity(*v) < n {
        unsafe {
            let ptr: **raw::VecRepr = cast::transmute(v);
            rustrt::vec_reserve_shared(sys::get_type_desc::<T>(),
                                       ptr, n as size_t);
        }
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
fn reserve_at_least<T>(v: &mut ~[T], n: uint) {
    reserve(v, uint::next_power_of_two(n));
}

/// Returns the number of elements the vector can hold without reallocating
#[inline(always)]
pure fn capacity<T>(&&v: ~[const T]) -> uint {
    unsafe {
        let repr: **raw::VecRepr = ::cast::reinterpret_cast(&addr_of(v));
        (**repr).unboxed.alloc / sys::size_of::<T>()
    }
}

/// Returns the length of a vector
#[inline(always)]
pure fn len<T>(&&v: &[const T]) -> uint {
    as_const_buf(v, |_p, len| len)
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pure fn from_fn<T>(n_elts: uint, op: iter::InitOp<T>) -> ~[T] {
    let mut v = with_capacity(n_elts);
    let mut i: uint = 0u;
    while i < n_elts unsafe { raw::set(v, i, op(i)); i += 1u; }
    unsafe { raw::set_len(v, n_elts); }
    move v
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pure fn from_elem<T: Copy>(n_elts: uint, t: T) -> ~[T] {
    let mut v = with_capacity(n_elts);
    let mut i: uint = 0u;
    unsafe { // because unsafe::set is unsafe
        while i < n_elts { raw::set(v, i, t); i += 1u; }
        unsafe { raw::set_len(v, n_elts); }
    }
    move v
}

/// Creates a new unique vector with the same contents as the slice
pure fn from_slice<T: Copy>(t: &[T]) -> ~[T] {
    from_fn(t.len(), |i| t[i])
}

pure fn with_capacity<T>(capacity: uint) -> ~[T] {
    let mut vec = ~[];
    unsafe { reserve(&mut vec, capacity); }
    return move vec;
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial size for the vector.
 *
 * # Arguments
 *
 * * size - An initial size of the vector to reserve
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pure fn build_sized<A>(size: uint,
                       builder: fn(push: pure fn(+v: A))) -> ~[A] {
    let mut vec = with_capacity(size);
    builder(|+x| unsafe { vec.push(move x) });
    move vec
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 *
 * # Arguments
 *
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pure fn build<A>(builder: fn(push: pure fn(+v: A))) -> ~[A] {
    build_sized(4, builder)
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial size for the vector.
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the vector to reserve
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pure fn build_sized_opt<A>(size: Option<uint>,
                           builder: fn(push: pure fn(+v: A))) -> ~[A] {
    build_sized(size.get_default(4), builder)
}

/// Produces a mut vector from an immutable vector.
pure fn to_mut<T>(+v: ~[T]) -> ~[mut T] {
    unsafe { ::cast::transmute(move v) }
}

/// Produces an immutable vector from a mut vector.
pure fn from_mut<T>(+v: ~[mut T]) -> ~[T] {
    unsafe { ::cast::transmute(move v) }
}

// Accessors

/// Returns the first element of a vector
pure fn head<T: Copy>(v: &[const T]) -> T { v[0] }

/// Returns a vector containing all but the first element of a slice
pure fn tail<T: Copy>(v: &[const T]) -> ~[T] {
    return slice(v, 1u, len(v));
}

/**
 * Returns a vector containing all but the first `n` \
 * elements of a slice
 */
pure fn tailn<T: Copy>(v: &[const T], n: uint) -> ~[T] {
    slice(v, n, len(v))
}

/// Returns a vector containing all but the last element of a slice
pure fn init<T: Copy>(v: &[const T]) -> ~[T] {
    assert len(v) != 0u;
    slice(v, 0u, len(v) - 1u)
}

/// Returns the last element of the slice `v`, failing if the slice is empty.
pure fn last<T: Copy>(v: &[const T]) -> T {
    if len(v) == 0u { fail ~"last_unsafe: empty vector" }
    v[len(v) - 1u]
}

/**
 * Returns `Some(x)` where `x` is the last element of the slice `v`,
 * or `none` if the vector is empty.
 */
pure fn last_opt<T: Copy>(v: &[const T]) -> Option<T> {
    if len(v) == 0u { return None; }
    Some(v[len(v) - 1u])
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
pure fn slice<T: Copy>(v: &[const T], start: uint, end: uint) -> ~[T] {
    assert (start <= end);
    assert (end <= len(v));
    let mut result = ~[];
    unsafe {
        for uint::range(start, end) |i| { result.push(v[i]) }
    }
    move result
}

/// Return a slice that points into another slice.
pure fn view<T>(v: &[T], start: uint, end: uint) -> &[T] {
    assert (start <= end);
    assert (end <= len(v));
    do as_imm_buf(v) |p, _len| {
        unsafe {
            ::cast::reinterpret_cast(
                &(ptr::offset(p, start),
                  (end - start) * sys::size_of::<T>()))
        }
    }
}

/// Return a slice that points into another slice.
pure fn mut_view<T>(v: &[mut T], start: uint, end: uint) -> &[mut T] {
    assert (start <= end);
    assert (end <= len(v));
    do as_mut_buf(v) |p, _len| {
        unsafe {
            ::cast::reinterpret_cast(
                &(ptr::mut_offset(p, start),
                  (end - start) * sys::size_of::<T>()))
        }
    }
}

/// Return a slice that points into another slice.
pure fn const_view<T>(v: &[const T], start: uint, end: uint) -> &[const T] {
    assert (start <= end);
    assert (end <= len(v));
    do as_const_buf(v) |p, _len| {
        unsafe {
            ::cast::reinterpret_cast(
                &(ptr::const_offset(p, start),
                  (end - start) * sys::size_of::<T>()))
        }
    }
}

/// Split the vector `v` by applying each element against the predicate `f`.
fn split<T: Copy>(v: &[T], f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut result = ~[];
    while start < ln {
        match position_between(v, start, ln, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, start, i));
                start = i + 1u;
            }
        }
    }
    result.push(slice(v, start, ln));
    move result
}

/**
 * Split the vector `v` by applying each element against the predicate `f` up
 * to `n` times.
 */
fn splitn<T: Copy>(v: &[T], n: uint, f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut start = 0u;
    let mut count = n;
    let mut result = ~[];
    while start < ln && count > 0u {
        match position_between(v, start, ln, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, start, i));
                // Make sure to skip the separator.
                start = i + 1u;
                count -= 1u;
            }
        }
    }
    result.push(slice(v, start, ln));
    move result
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f`.
 */
fn rsplit<T: Copy>(v: &[T], f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut end = ln;
    let mut result = ~[];
    while end > 0u {
        match rposition_between(v, 0u, end, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, i + 1u, end));
                end = i;
            }
        }
    }
    result.push(slice(v, 0u, end));
    reverse(result);
    return move result;
}

/**
 * Reverse split the vector `v` by applying each element against the predicate
 * `f` up to `n times.
 */
fn rsplitn<T: Copy>(v: &[T], n: uint, f: fn(T) -> bool) -> ~[~[T]] {
    let ln = len(v);
    if (ln == 0u) { return ~[] }

    let mut end = ln;
    let mut count = n;
    let mut result = ~[];
    while end > 0u && count > 0u {
        match rposition_between(v, 0u, end, f) {
            None => break,
            Some(i) => {
                result.push(slice(v, i + 1u, end));
                // Make sure to skip the separator.
                end = i;
                count -= 1u;
            }
        }
    }
    result.push(slice(v, 0u, end));
    reverse(result);
    move result
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
            let vv = raw::to_ptr(vv);
            rr <- *vv;

            for uint::range(1, ln) |i| {
                let r <- *ptr::offset(vv, i);
                v.push(move r);
            }
        }
        raw::set_len(vv, 0);

        move rr
    }
}

/// Prepend an element to the vector
fn unshift<T>(&v: ~[T], +x: T) {
    let mut vv = ~[move x];
    v <-> vv;
    while len(vv) > 0 {
        v.push(shift(vv));
    }
}

fn consume<T>(+v: ~[T], f: fn(uint, +v: T)) unsafe {
    do as_imm_buf(v) |p, ln| {
        for uint::range(0, ln) |i| {
            let x <- *ptr::offset(p, i);
            f(i, move x);
        }
    }

    raw::set_len(v, 0);
}

fn consume_mut<T>(+v: ~[mut T], f: fn(uint, +v: T)) unsafe {
    do as_imm_buf(v) |p, ln| {
        for uint::range(0, ln) |i| {
            let x <- *ptr::offset(p, i);
            f(i, move x);
        }
    }

    raw::set_len(v, 0);
}

/// Remove the last element from a vector and return it
fn pop<T>(&v: ~[const T]) -> T {
    let ln = len(v);
    if ln == 0 {
        fail ~"sorry, cannot vec::pop an empty vector"
    }
    let valptr = ptr::mut_addr_of(v[ln - 1u]);
    unsafe {
        let val <- *valptr;
        raw::set_len(v, ln - 1u);
        move val
    }
}

/**
 * Remove an element from anywhere in the vector and return it, replacing it
 * with the last element. This does not preserve ordering, but is O(1).
 *
 * Fails if index >= length.
 */
fn swap_remove<T>(&v: ~[const T], index: uint) -> T {
    let ln = len(v);
    if index >= ln {
        fail fmt!("vec::swap_remove - index %u >= length %u", index, ln);
    }
    let lastptr = ptr::mut_addr_of(v[ln - 1]);
    unsafe {
        let mut val <- *lastptr;
        if index < ln - 1 {
            let valptr = ptr::mut_addr_of(v[index]);
            *valptr <-> val;
        }
        raw::set_len(v, ln - 1);
        move val
    }
}

/// Append an element to a vector
#[inline(always)]
fn push<T>(v: &mut ~[T], +initval: T) {
    unsafe {
        let repr: **raw::VecRepr = ::cast::transmute(copy v);
        let fill = (**repr).unboxed.fill;
        if (**repr).unboxed.alloc > fill {
            push_fast(v, move initval);
        }
        else {
            push_slow(v, move initval);
        }
    }
}

// This doesn't bother to make sure we have space.
#[inline(always)] // really pretty please
unsafe fn push_fast<T>(+v: &mut ~[T], +initval: T) {
    let repr: **raw::VecRepr = ::cast::transmute(v);
    let fill = (**repr).unboxed.fill;
    (**repr).unboxed.fill += sys::size_of::<T>();
    let p = ptr::addr_of((**repr).unboxed.data);
    let p = ptr::offset(p, fill) as *mut T;
    rusti::move_val_init(*p, move initval);
}

#[inline(never)]
fn push_slow<T>(+v: &mut ~[T], +initval: T) {
    reserve_at_least(v, v.len() + 1u);
    unsafe { push_fast(v, move initval) }
}

#[inline(always)]
fn push_all<T: Copy>(+v: &mut ~[T], rhs: &[const T]) {
    reserve(v, v.len() + rhs.len());

    for uint::range(0u, rhs.len()) |i| {
        push(v, unsafe { raw::get(rhs, i) })
    }
}

#[inline(always)]
fn push_all_move<T>(v: &mut ~[T], -rhs: ~[const T]) {
    reserve(v, v.len() + rhs.len());
    unsafe {
        do as_imm_buf(rhs) |p, len| {
            for uint::range(0, len) |i| {
                let x <- *ptr::offset(p, i);
                push(v, move x);
            }
        }
        raw::set_len(rhs, 0);
    }
}

/// Shorten a vector, dropping excess elements.
fn truncate<T>(&v: ~[const T], newlen: uint) {
    do as_imm_buf(v) |p, oldlen| {
        assert(newlen <= oldlen);
        unsafe {
            // This loop is optimized out for non-drop types.
            for uint::range(newlen, oldlen) |i| {
                let _dropped <- *ptr::offset(p, i);
            }
            raw::set_len(v, newlen);
        }
    }
}

/**
 * Remove consecutive repeated elements from a vector; if the vector is
 * sorted, this removes all duplicates.
 */
fn dedup<T: Eq>(&v: ~[const T]) unsafe {
    if v.len() < 1 { return; }
    let mut last_written = 0, next_to_read = 1;
    do as_const_buf(v) |p, ln| {
        // We have a mutable reference to v, so we can make arbitrary changes.
        // (cf. push and pop)
        let p = p as *mut T;
        // last_written < next_to_read <= ln
        while next_to_read < ln {
            // last_written < next_to_read < ln
            if *ptr::mut_offset(p, next_to_read) ==
                *ptr::mut_offset(p, last_written) {
                let _dropped <- *ptr::mut_offset(p, next_to_read);
            } else {
                last_written += 1;
                // last_written <= next_to_read < ln
                if next_to_read != last_written {
                    *ptr::mut_offset(p, last_written) <-
                        *ptr::mut_offset(p, next_to_read);
                }
            }
            // last_written <= next_to_read < ln
            next_to_read += 1;
            // last_written < next_to_read <= ln
        }
    }
    // last_written < next_to_read == ln
    raw::set_len(v, last_written + 1);
}


// Appending
#[inline(always)]
pure fn append<T: Copy>(+lhs: ~[T], rhs: &[const T]) -> ~[T] {
    let mut v <- lhs;
    unsafe {
        v.push_all(rhs);
    }
    move v
}

#[inline(always)]
pure fn append_one<T>(+lhs: ~[T], +x: T) -> ~[T] {
    let mut v <- lhs;
    unsafe { v.push(move x); }
    move v
}

#[inline(always)]
pure fn append_mut<T: Copy>(+lhs: ~[mut T], rhs: &[const T]) -> ~[mut T] {
    to_mut(append(from_mut(lhs), rhs))
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
fn grow<T: Copy>(&v: ~[T], n: uint, initval: T) {
    reserve_at_least(&mut v, len(v) + n);
    let mut i: uint = 0u;

    while i < n {
        v.push(initval);
        i += 1u;
    }
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
fn grow_fn<T>(&v: ~[T], n: uint, op: iter::InitOp<T>) {
    reserve_at_least(&mut v, len(v) + n);
    let mut i: uint = 0u;
    while i < n {
        v.push(op(i));
        i += 1u;
    }
}

/**
 * Sets the value of a vector element at a given index, growing the vector as
 * needed
 *
 * Sets the element at position `index` to `val`. If `index` is past the end
 * of the vector, expands the vector by replicating `initval` to fill the
 * intervening space.
 */
fn grow_set<T: Copy>(&v: ~[T], index: uint, initval: T, val: T) {
    if index >= len(v) { grow(v, index - len(v) + 1u, initval); }
    v[index] = val;
}

// Functional utilities

/// Apply a function to each element of a vector and return the results
pure fn map<T, U>(v: &[T], f: fn(v: &T) -> U) -> ~[U] {
    let mut result = with_capacity(len(v));
    for each(v) |elem| {
        unsafe {
            result.push(f(elem));
        }
    }
    move result
}

fn map_consume<T, U>(+v: ~[T], f: fn(+v: T) -> U) -> ~[U] {
    let mut result = ~[];
    do consume(move v) |_i, x| {
        result.push(f(move x));
    }
    move result
}

/// Apply a function to each element of a vector and return the results
pure fn mapi<T, U>(v: &[T], f: fn(uint, v: &T) -> U) -> ~[U] {
    let mut i = 0;
    do map(v) |e| {
        i += 1;
        f(i - 1, e)
    }
}

/**
 * Apply a function to each element of a vector and return a concatenation
 * of each result vector
 */
pure fn flat_map<T, U>(v: &[T], f: fn(T) -> ~[U]) -> ~[U] {
    let mut result = ~[];
    for each(v) |elem| { unsafe{ result.push_all_move(f(*elem)); } }
    move result
}

/// Apply a function to each pair of elements and return the results
pure fn map2<T: Copy, U: Copy, V>(v0: &[T], v1: &[U],
                                  f: fn(T, U) -> V) -> ~[V] {
    let v0_len = len(v0);
    if v0_len != len(v1) { fail; }
    let mut u: ~[V] = ~[];
    let mut i = 0u;
    while i < v0_len {
        unsafe { u.push(f(copy v0[i], copy v1[i])) };
        i += 1u;
    }
    move u
}

/**
 * Apply a function to each element of a vector and return the results
 *
 * If function `f` returns `none` then that element is excluded from
 * the resulting vector.
 */
pure fn filter_map<T, U: Copy>(v: &[T], f: fn(T) -> Option<U>)
    -> ~[U] {
    let mut result = ~[];
    for each(v) |elem| {
        match f(*elem) {
          None => {/* no-op */ }
          Some(result_elem) => unsafe { result.push(result_elem); }
        }
    }
    move result
}

/**
 * Construct a new vector from the elements of a vector for which some
 * predicate holds.
 *
 * Apply function `f` to each element of `v` and return a vector containing
 * only those elements for which `f` returned true.
 */
pure fn filter<T: Copy>(v: &[T], f: fn(T) -> bool) -> ~[T] {
    let mut result = ~[];
    for each(v) |elem| {
        if f(*elem) { unsafe { result.push(*elem); } }
    }
    move result
}

/**
 * Concatenate a vector of vectors.
 *
 * Flattens a vector of vectors of T into a single vector of T.
 */
pure fn concat<T: Copy>(v: &[~[T]]) -> ~[T] {
    let mut r = ~[];
    for each(v) |inner| { unsafe { r.push_all(*inner); } }
    move r
}

/// Concatenate a vector of vectors, placing a given separator between each
pure fn connect<T: Copy>(v: &[~[T]], sep: T) -> ~[T] {
    let mut r: ~[T] = ~[];
    let mut first = true;
    for each(v) |inner| {
        if first { first = false; } else { unsafe { r.push(sep); } }
        unsafe { r.push_all(*inner) };
    }
    move r
}

/// Reduce a vector from left to right
pure fn foldl<T: Copy, U>(z: T, v: &[U], p: fn(T, U) -> T) -> T {
    let mut accum = z;
    for each(v) |elt| {
        accum = p(accum, *elt);
    }
    return accum;
}

/// Reduce a vector from right to left
pure fn foldr<T, U: Copy>(v: &[T], z: U, p: fn(T, U) -> U) -> U {
    let mut accum = z;
    for rev_each(v) |elt| {
        accum = p(*elt, accum);
    }
    return accum;
}

/**
 * Return true if a predicate matches any elements
 *
 * If the vector contains no elements then false is returned.
 */
pure fn any<T>(v: &[T], f: fn(T) -> bool) -> bool {
    for each(v) |elem| { if f(*elem) { return true; } }
    return false;
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
        if f(v0[i], v1[i]) { return true; };
        i += 1u;
    }
    return false;
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pure fn all<T>(v: &[T], f: fn(T) -> bool) -> bool {
    for each(v) |elem| { if !f(*elem) { return false; } }
    return true;
}

/**
 * Return true if a predicate matches all elements
 *
 * If the vector contains no elements then true is returned.
 */
pure fn alli<T>(v: &[T], f: fn(uint, T) -> bool) -> bool {
    for eachi(v) |i, elem| { if !f(i, *elem) { return false; } }
    return true;
}

/**
 * Return true if a predicate matches all elements in both vectors.
 *
 * If the vectors are not the same size then false is returned.
 */
pure fn all2<T, U>(v0: &[T], v1: &[U],
                   f: fn(T, U) -> bool) -> bool {
    let v0_len = len(v0);
    if v0_len != len(v1) { return false; }
    let mut i = 0u;
    while i < v0_len { if !f(v0[i], v1[i]) { return false; }; i += 1u; }
    return true;
}

/// Return true if a vector contains an element with the given value
pure fn contains<T: Eq>(v: &[T], x: T) -> bool {
    for each(v) |elt| { if x == *elt { return true; } }
    return false;
}

/// Returns the number of elements that are equal to a given value
pure fn count<T: Eq>(v: &[T], x: T) -> uint {
    let mut cnt = 0u;
    for each(v) |elt| { if x == *elt { cnt += 1u; } }
    return cnt;
}

/**
 * Search for the first element that matches a given predicate
 *
 * Apply function `f` to each element of `v`, starting from the first.
 * When function `f` returns true then an option containing the element
 * is returned. If `f` matches no elements then none is returned.
 */
pure fn find<T: Copy>(v: &[T], f: fn(T) -> bool) -> Option<T> {
    find_between(v, 0u, len(v), f)
}

/**
 * Search for the first element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is returned.
 */
pure fn find_between<T: Copy>(v: &[T], start: uint, end: uint,
                      f: fn(T) -> bool) -> Option<T> {
    position_between(v, start, end, f).map(|i| v[i])
}

/**
 * Search for the last element that matches a given predicate
 *
 * Apply function `f` to each element of `v` in reverse order. When function
 * `f` returns true then an option containing the element is returned. If `f`
 * matches no elements then none is returned.
 */
pure fn rfind<T: Copy>(v: &[T], f: fn(T) -> bool) -> Option<T> {
    rfind_between(v, 0u, len(v), f)
}

/**
 * Search for the last element that matches a given predicate within a range
 *
 * Apply function `f` to each element of `v` in reverse order within the range
 * [`start`, `end`). When function `f` returns true then an option containing
 * the element is returned. If `f` matches no elements then none is returned.
 */
pure fn rfind_between<T: Copy>(v: &[T], start: uint, end: uint,
                               f: fn(T) -> bool) -> Option<T> {
    rposition_between(v, start, end, f).map(|i| v[i])
}

/// Find the first index containing a matching value
pure fn position_elem<T: Eq>(v: &[T], x: T) -> Option<uint> {
    position(v, |y| x == y)
}

/**
 * Find the first index matching some predicate
 *
 * Apply function `f` to each element of `v`.  When function `f` returns true
 * then an option containing the index is returned. If `f` matches no elements
 * then none is returned.
 */
pure fn position<T>(v: &[T], f: fn(T) -> bool) -> Option<uint> {
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
                            f: fn(T) -> bool) -> Option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = start;
    while i < end { if f(v[i]) { return Some::<uint>(i); } i += 1u; }
    return None;
}

/// Find the last index containing a matching value
pure fn rposition_elem<T: Eq>(v: &[T], x: T) -> Option<uint> {
    rposition(v, |y| x == y)
}

/**
 * Find the last index matching some predicate
 *
 * Apply function `f` to each element of `v` in reverse order.  When function
 * `f` returns true then an option containing the index is returned. If `f`
 * matches no elements then none is returned.
 */
pure fn rposition<T>(v: &[T], f: fn(T) -> bool) -> Option<uint> {
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
                             f: fn(T) -> bool) -> Option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = end;
    while i > start {
        if f(v[i - 1u]) { return Some::<uint>(i - 1u); }
        i -= 1u;
    }
    return None;
}

// FIXME: if issue #586 gets implemented, could have a postcondition
// saying the two result lists have the same length -- or, could
// return a nominal record with a constraint saying that, instead of
// returning a tuple (contingent on issue #869)
/**
 * Convert a vector of pairs into a pair of vectors, by reference. As unzip().
 */
pure fn unzip_slice<T: Copy, U: Copy>(v: &[(T, U)]) -> (~[T], ~[U]) {
    let mut ts = ~[], us = ~[];
    for each(v) |p| {
        let (t, u) = *p;
        unsafe {
            ts.push(t);
            us.push(u);
        }
    }
    return (move ts, move us);
}

/**
 * Convert a vector of pairs into a pair of vectors.
 *
 * Returns a tuple containing two vectors where the i-th element of the first
 * vector contains the first element of the i-th tuple of the input vector,
 * and the i-th element of the second vector contains the second element
 * of the i-th tuple of the input vector.
 */
pure fn unzip<T,U>(+v: ~[(T, U)]) -> (~[T], ~[U]) {
    let mut ts = ~[], us = ~[];
    unsafe {
        do consume(move v) |_i, p| {
            let (t, u) = move p;
            ts.push(move t);
            us.push(move u);
        }
    }
    (move ts, move us)
}

/**
 * Convert two vectors to a vector of pairs, by reference. As zip().
 */
pure fn zip_slice<T: Copy, U: Copy>(v: &[const T], u: &[const U])
        -> ~[(T, U)] {
    let mut zipped = ~[];
    let sz = len(v);
    let mut i = 0u;
    assert sz == len(u);
    while i < sz unsafe { zipped.push((v[i], u[i])); i += 1u; }
    move zipped
}

/**
 * Convert two vectors to a vector of pairs.
 *
 * Returns a vector of tuples, where the i-th tuple contains contains the
 * i-th elements from each of the input vectors.
 */
pure fn zip<T, U>(+v: ~[const T], +u: ~[const U]) -> ~[(T, U)] {
    let mut v = move v, u = move u, i = len(v);
    assert i == len(u);
    let mut w = with_capacity(i);
    while i > 0 {
        unsafe { w.push((pop(v),pop(u))); }
        i -= 1;
    }
    unsafe { reverse(w); }
    move w
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
fn swap<T>(v: &[mut T], a: uint, b: uint) {
    v[a] <-> v[b];
}

/// Reverse the order of elements in a vector, in place
fn reverse<T>(v: &[mut T]) {
    let mut i: uint = 0u;
    let ln = len::<T>(v);
    while i < ln / 2u { v[i] <-> v[ln - i - 1u]; i += 1u; }
}

/// Returns a vector with the order of elements reversed
pure fn reversed<T: Copy>(v: &[const T]) -> ~[T] {
    let mut rs: ~[T] = ~[];
    let mut i = len::<T>(v);
    if i == 0 { return (move rs); } else { i -= 1; }
    unsafe {
        while i != 0 { rs.push(v[i]); i -= 1; }
        rs.push(v[0]);
    }
    move rs
}

/**
 * Iterates over a vector, with option to break
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pure fn each<T>(v: &r/[T], f: fn((&r/T)) -> bool) {
    //             ^^^^
    // NB---this CANNOT be &[const T]!  The reason
    // is that you are passing it to `f()` using
    // an immutable.

    do vec::as_imm_buf(v) |p, n| {
        let mut n = n;
        let mut p = p;
        while n > 0u {
            unsafe {
                let q = cast::copy_lifetime_vec(v, &*p);
                if !f(q) { break; }
                p = ptr::offset(p, 1u);
            }
            n -= 1u;
        }
    }
}

/// Like `each()`, but for the case where you have
/// a vector with mutable contents and you would like
/// to mutate the contents as you iterate.
#[inline(always)]
fn each_mut<T>(v: &[mut T], f: fn(elem: &mut T) -> bool) {
    let mut i = 0;
    let n = v.len();
    while i < n {
        if !f(&mut v[i]) {
            return;
        }
        i += 1;
    }
}

/// Like `each()`, but for the case where you have a vector that *may or may
/// not* have mutable contents.
#[inline(always)]
pure fn each_const<T>(v: &[const T], f: fn(elem: &const T) -> bool) {
    let mut i = 0;
    let n = v.len();
    while i < n {
        if !f(&const v[i]) {
            return;
        }
        i += 1;
    }
}

/**
 * Iterates over a vector's elements and indices
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pure fn eachi<T>(v: &r/[T], f: fn(uint, v: &r/T) -> bool) {
    let mut i = 0;
    for each(v) |p| {
        if !f(i, p) { return; }
        i += 1;
    }
}

/**
 * Iterates over a vector's elements in reverse
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pure fn rev_each<T>(v: &r/[T], blk: fn(v: &r/T) -> bool) {
    rev_eachi(v, |_i, v| blk(v))
}

/**
 * Iterates over a vector's elements and indices in reverse
 *
 * Return true to continue, false to break.
 */
#[inline(always)]
pure fn rev_eachi<T>(v: &r/[T], blk: fn(i: uint, v: &r/T) -> bool) {
    let mut i = v.len();
    while i > 0 {
        i -= 1;
        if !blk(i, &v[i]) {
            return;
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
 * Iterate over all permutations of vector `v`.
 *
 * Permutations are produced in lexicographic order with respect to the order
 * of elements in `v` (so if `v` is sorted then the permutations are
 * lexicographically sorted).
 *
 * The total number of permutations produced is `len(v)!`.  If `v` contains
 * repeated elements, then some permutations are repeated.
 */
pure fn permute<T: Copy>(v: &[const T], put: fn(~[T])) {
    let ln = len(v);
    if ln == 0u {
        put(~[]);
    } else {
        let mut i = 0u;
        while i < ln {
            let elt = v[i];
            let mut rest = slice(v, 0u, i);
            unsafe {
                rest.push_all(const_view(v, i+1u, ln));
                permute(rest, |permutation| {
                    put(append(~[elt], permutation))
                })
            }
            i += 1u;
        }
    }
}

pure fn windowed<TT: Copy>(nn: uint, xx: &[TT]) -> ~[~[TT]] {
    let mut ww = ~[];
    assert 1u <= nn;
    for vec::eachi (xx) |ii, _x| {
        let len = vec::len(xx);
        if ii+nn <= len unsafe {
            ww.push(vec::slice(xx, ii, ii+nn));
        }
    }
    move ww
}

/**
 * Work with the buffer of a vector.
 *
 * Allows for unsafe manipulation of vector contents, which is useful for
 * foreign interop.
 */
#[inline(always)]
pure fn as_imm_buf<T,U>(s: &[T], /* NB---this CANNOT be const, see below */
                        f: fn(*T, uint) -> U) -> U {

    // NB---Do not change the type of s to `&[const T]`.  This is
    // unsound.  The reason is that we are going to create immutable pointers
    // into `s` and pass them to `f()`, but in fact they are potentially
    // pointing at *mutable memory*.  Use `as_const_buf` or `as_mut_buf`
    // instead!

    unsafe {
        let v : *(*T,uint) =
            ::cast::reinterpret_cast(&ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len / sys::size_of::<T>())
    }
}

/// Similar to `as_imm_buf` but passing a `*const T`
#[inline(always)]
pure fn as_const_buf<T,U>(s: &[const T],
                          f: fn(*const T, uint) -> U) -> U {

    unsafe {
        let v : *(*const T,uint) =
            ::cast::reinterpret_cast(&ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len / sys::size_of::<T>())
    }
}

/// Similar to `as_imm_buf` but passing a `*mut T`
#[inline(always)]
pure fn as_mut_buf<T,U>(s: &[mut T],
                        f: fn(*mut T, uint) -> U) -> U {

    unsafe {
        let v : *(*mut T,uint) =
            ::cast::reinterpret_cast(&ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len / sys::size_of::<T>())
    }
}

// Equality

pure fn eq<T: Eq>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    if a_len != b_len { return false; }

    let mut i = 0;
    while i < a_len {
        if a[i] != b[i] { return false; }
        i += 1;
    }

    return true;
}

impl<T: Eq> &[T] : Eq {
    #[inline(always)]
    pure fn eq(other: & &[T]) -> bool { eq(self, (*other)) }
    #[inline(always)]
    pure fn ne(other: & &[T]) -> bool { !self.eq(other) }
}

impl<T: Eq> ~[T] : Eq {
    #[inline(always)]
    pure fn eq(other: &~[T]) -> bool { eq(self, (*other)) }
    #[inline(always)]
    pure fn ne(other: &~[T]) -> bool { !self.eq(other) }
}

impl<T: Eq> @[T] : Eq {
    #[inline(always)]
    pure fn eq(other: &@[T]) -> bool { eq(self, (*other)) }
    #[inline(always)]
    pure fn ne(other: &@[T]) -> bool { !self.eq(other) }
}

// Lexicographical comparison

pure fn lt<T: Ord>(a: &[T], b: &[T]) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    let mut end = uint::min(a_len, b_len);

    let mut i = 0;
    while i < end {
        let (c_a, c_b) = (&a[i], &b[i]);
        if *c_a < *c_b { return true; }
        if *c_a > *c_b { return false; }
        i += 1;
    }

    return a_len < b_len;
}

pure fn le<T: Ord>(a: &[T], b: &[T]) -> bool { !lt(b, a) }
pure fn ge<T: Ord>(a: &[T], b: &[T]) -> bool { !lt(a, b) }
pure fn gt<T: Ord>(a: &[T], b: &[T]) -> bool { lt(b, a)  }

impl<T: Ord> &[T] : Ord {
    #[inline(always)]
    pure fn lt(other: & &[T]) -> bool { lt(self, (*other)) }
    #[inline(always)]
    pure fn le(other: & &[T]) -> bool { le(self, (*other)) }
    #[inline(always)]
    pure fn ge(other: & &[T]) -> bool { ge(self, (*other)) }
    #[inline(always)]
    pure fn gt(other: & &[T]) -> bool { gt(self, (*other)) }
}

impl<T: Ord> ~[T] : Ord {
    #[inline(always)]
    pure fn lt(other: &~[T]) -> bool { lt(self, (*other)) }
    #[inline(always)]
    pure fn le(other: &~[T]) -> bool { le(self, (*other)) }
    #[inline(always)]
    pure fn ge(other: &~[T]) -> bool { ge(self, (*other)) }
    #[inline(always)]
    pure fn gt(other: &~[T]) -> bool { gt(self, (*other)) }
}

impl<T: Ord> @[T] : Ord {
    #[inline(always)]
    pure fn lt(other: &@[T]) -> bool { lt(self, (*other)) }
    #[inline(always)]
    pure fn le(other: &@[T]) -> bool { le(self, (*other)) }
    #[inline(always)]
    pure fn ge(other: &@[T]) -> bool { ge(self, (*other)) }
    #[inline(always)]
    pure fn gt(other: &@[T]) -> bool { gt(self, (*other)) }
}

#[cfg(notest)]
mod traits {
    #[legacy_exports];
    impl<T: Copy> ~[T] : Add<&[const T],~[T]> {
        #[inline(always)]
        pure fn add(rhs: & &[const T]) -> ~[T] {
            append(copy self, (*rhs))
        }
    }

    impl<T: Copy> ~[mut T] : Add<&[const T],~[mut T]> {
        #[inline(always)]
        pure fn add(rhs: & &[const T]) -> ~[mut T] {
            append_mut(copy self, (*rhs))
        }
    }
}

#[cfg(test)]
mod traits {
    #[legacy_exports];}

trait ConstVector {
    pure fn is_empty() -> bool;
    pure fn is_not_empty() -> bool;
    pure fn len() -> uint;
}

/// Extension methods for vectors
impl<T> &[const T]: ConstVector {
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

trait CopyableVector<T> {
    pure fn head() -> T;
    pure fn init() -> ~[T];
    pure fn last() -> T;
    pure fn slice(start: uint, end: uint) -> ~[T];
    pure fn tail() -> ~[T];
}

/// Extension methods for vectors
impl<T: Copy> &[const T]: CopyableVector<T> {
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

trait ImmutableVector<T> {
    pure fn foldr<U: Copy>(z: U, p: fn(T, U) -> U) -> U;
    pure fn map<U>(f: fn(v: &T) -> U) -> ~[U];
    pure fn mapi<U>(f: fn(uint, v: &T) -> U) -> ~[U];
    fn map_r<U>(f: fn(x: &T) -> U) -> ~[U];
    pure fn alli(f: fn(uint, T) -> bool) -> bool;
    pure fn flat_map<U>(f: fn(T) -> ~[U]) -> ~[U];
    pure fn filter_map<U: Copy>(f: fn(T) -> Option<U>) -> ~[U];
}

trait ImmutableEqVector<T: Eq> {
    pure fn position(f: fn(T) -> bool) -> Option<uint>;
    pure fn position_elem(x: T) -> Option<uint>;
    pure fn rposition(f: fn(T) -> bool) -> Option<uint>;
    pure fn rposition_elem(x: T) -> Option<uint>;
}

/// Extension methods for vectors
impl<T> &[T]: ImmutableVector<T> {
    /// Reduce a vector from right to left
    #[inline]
    pure fn foldr<U: Copy>(z: U, p: fn(T, U) -> U) -> U { foldr(self, z, p) }
    /// Apply a function to each element of a vector and return the results
    #[inline]
    pure fn map<U>(f: fn(v: &T) -> U) -> ~[U] { map(self, f) }
    /**
     * Apply a function to the index and value of each element in the vector
     * and return the results
     */
    pure fn mapi<U>(f: fn(uint, v: &T) -> U) -> ~[U] {
        mapi(self, f)
    }

    #[inline]
    fn map_r<U>(f: fn(x: &T) -> U) -> ~[U] {
        let mut r = ~[];
        let mut i = 0;
        while i < self.len() {
            r.push(f(&self[i]));
            i += 1;
        }
        move r
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
    pure fn filter_map<U: Copy>(f: fn(T) -> Option<U>) -> ~[U] {
        filter_map(self, f)
    }
}

impl<T: Eq> &[T]: ImmutableEqVector<T> {
    /**
     * Find the first index matching some predicate
     *
     * Apply function `f` to each element of `v`.  When function `f` returns
     * true then an option containing the index is returned. If `f` matches no
     * elements then none is returned.
     */
    #[inline]
    pure fn position(f: fn(T) -> bool) -> Option<uint> { position(self, f) }
    /// Find the first index containing a matching value
    #[inline]
    pure fn position_elem(x: T) -> Option<uint> { position_elem(self, x) }
    /**
     * Find the last index matching some predicate
     *
     * Apply function `f` to each element of `v` in reverse order.  When
     * function `f` returns true then an option containing the index is
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    pure fn rposition(f: fn(T) -> bool) -> Option<uint> { rposition(self, f) }
    /// Find the last index containing a matching value
    #[inline]
    pure fn rposition_elem(x: T) -> Option<uint> { rposition_elem(self, x) }
}

trait ImmutableCopyableVector<T> {
    pure fn filter(f: fn(T) -> bool) -> ~[T];
    pure fn rfind(f: fn(T) -> bool) -> Option<T>;
}

/// Extension methods for vectors
impl<T: Copy> &[T]: ImmutableCopyableVector<T> {
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
     * Search for the last element that matches a given predicate
     *
     * Apply function `f` to each element of `v` in reverse order. When
     * function `f` returns true then an option containing the element is
     * returned. If `f` matches no elements then none is returned.
     */
    #[inline]
    pure fn rfind(f: fn(T) -> bool) -> Option<T> { rfind(self, f) }
}

trait MutableVector<T> {
    fn push(&mut self, +t: T);
    fn push_all_move(&mut self, -rhs: ~[const T]);
}

trait MutableCopyableVector<T: Copy> {
    fn push_all(&mut self, rhs: &[const T]);
}

impl<T> ~[T]: MutableVector<T> {
    fn push(&mut self, +t: T) {
        push(self, move t);
    }

    fn push_all_move(&mut self, -rhs: ~[const T]) {
        push_all_move(self, move rhs);
    }
}

impl<T: Copy> ~[T]: MutableCopyableVector<T> {
    fn push_all(&mut self, rhs: &[const T]) {
        push_all(self, rhs);
    }
}

/// Unsafe operations
mod raw {
    #[legacy_exports];
    // FIXME: This should have crate visibility (#1893 blocks that)

    /// The internal representation of a (boxed) vector
    struct VecRepr {
        box_header: box::raw::BoxHeaderRepr,
        unboxed: UnboxedVecRepr
    }

    /// The internal 'unboxed' representation of a vector
    struct UnboxedVecRepr {
        mut fill: uint,
        mut alloc: uint,
        data: u8
    }

    type SliceRepr = {
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
        let mut dst = with_capacity(elts);
        set_len(dst, elts);
        as_mut_buf(dst, |p_dst, _len_dst| ptr::memcpy(p_dst, ptr, elts));
        move dst
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
        let repr: **VecRepr = ::cast::reinterpret_cast(&addr_of(v));
        (**repr).unboxed.fill = new_len * sys::size_of::<T>();
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
    unsafe fn to_ptr<T>(v: &[T]) -> *T {
        let repr: **SliceRepr = ::cast::reinterpret_cast(&addr_of(v));
        return ::cast::reinterpret_cast(&addr_of((**repr).data));
    }

    /** see `to_ptr()` */
    #[inline(always)]
    unsafe fn to_const_ptr<T>(v: &[const T]) -> *const T {
        let repr: **SliceRepr = ::cast::reinterpret_cast(&addr_of(v));
        return ::cast::reinterpret_cast(&addr_of((**repr).data));
    }

    /** see `to_ptr()` */
    #[inline(always)]
    unsafe fn to_mut_ptr<T>(v: &[mut T]) -> *mut T {
        let repr: **SliceRepr = ::cast::reinterpret_cast(&addr_of(v));
        return ::cast::reinterpret_cast(&addr_of((**repr).data));
    }

    /**
     * Form a slice from a pointer and length (as a number of units,
     * not bytes).
     */
    #[inline(always)]
    unsafe fn form_slice<T,U>(p: *T, len: uint, f: fn(v: &[T]) -> U) -> U {
        let pair = (p, len * sys::size_of::<T>());
        let v : *(&blk/[T]) =
            ::cast::reinterpret_cast(&ptr::addr_of(pair));
        f(*v)
    }

    /**
     * Unchecked vector indexing.
     */
    #[inline(always)]
    unsafe fn get<T: Copy>(v: &[const T], i: uint) -> T {
        as_const_buf(v, |p, _len| *ptr::const_offset(p, i))
    }

    /**
     * Unchecked vector index assignment.
     */
    #[inline(always)]
    unsafe fn set<T>(v: &[mut T], i: uint, +val: T) {
        let mut box = Some(move val);
        do as_mut_buf(v) |p, _len| {
            let mut box2 = None;
            box2 <-> box;
            rusti::move_val_init(*ptr::mut_offset(p, i),
                                 option::unwrap(move box2));
        }
    }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may overlap.
      */
    unsafe fn memcpy<T>(dst: &[mut T], src: &[const T], count: uint) {
        do as_mut_buf(dst) |p_dst, _len_dst| {
            do as_const_buf(src) |p_src, _len_src| {
                ptr::memcpy(p_dst, p_src, count)
            }
        }
    }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may overlap.
      */
    unsafe fn memmove<T>(dst: &[mut T], src: &[const T], count: uint) {
        do as_mut_buf(dst) |p_dst, _len_dst| {
            do as_const_buf(src) |p_src, _len_src| {
                ptr::memmove(p_dst, p_src, count)
            }
        }
    }
}

/// Operations on `[u8]`
mod bytes {
    #[legacy_exports];
    export cmp;
    export lt, le, eq, ne, ge, gt;
    export memcpy, memmove;

    /// Bytewise string comparison
    pure fn cmp(a: &~[u8], b: &~[u8]) -> int {
        let a_len = len(*a);
        let b_len = len(*b);
        let n = uint::min(a_len, b_len) as libc::size_t;
        let r = unsafe {
            libc::memcmp(raw::to_ptr(*a) as *libc::c_void,
                         raw::to_ptr(*b) as *libc::c_void, n) as int
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
    pure fn lt(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) < 0 }

    /// Bytewise less than or equal
    pure fn le(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) <= 0 }

    /// Bytewise equality
    pure fn eq(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) == 0 }

    /// Bytewise inequality
    pure fn ne(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) != 0 }

    /// Bytewise greater than or equal
    pure fn ge(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) >= 0 }

    /// Bytewise greater than
    pure fn gt(a: &~[u8], b: &~[u8]) -> bool { cmp(a, b) > 0 }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may not overlap.
      */
    fn memcpy(dst: &[mut u8], src: &[const u8], count: uint) {
        assert dst.len() >= count;
        assert src.len() >= count;

        unsafe { vec::raw::memcpy(dst, src, count) }
    }

    /**
      * Copies data from one vector to another.
      *
      * Copies `count` bytes from `src` to `dst`. The source and destination
      * may overlap.
      */
    fn memmove(dst: &[mut u8], src: &[const u8], count: uint) {
        assert dst.len() >= count;
        assert src.len() >= count;

        unsafe { vec::raw::memmove(dst, src, count) }
    }
}

// ___________________________________________________________________________
// ITERATION TRAIT METHODS
//
// This cannot be used with iter-trait.rs because of the region pointer
// required in the slice.

impl<A> &[A]: iter::BaseIter<A> {
    pure fn each(blk: fn(v: &A) -> bool) {
        // FIXME(#2263)---should be able to call each(self, blk)
        for each(self) |e| {
            if (!blk(e)) {
                return;
            }
        }
    }
    pure fn size_hint() -> Option<uint> { Some(len(self)) }
}

impl<A> &[A]: iter::ExtendedIter<A> {
    pure fn eachi(blk: fn(uint, v: &A) -> bool) { iter::eachi(&self, blk) }
    pure fn all(blk: fn(A) -> bool) -> bool { iter::all(self, blk) }
    pure fn any(blk: fn(A) -> bool) -> bool { iter::any(self, blk) }
    pure fn foldl<B>(+b0: B, blk: fn(B, A) -> B) -> B {
        iter::foldl(self, move b0, blk)
    }
    pure fn position(f: fn(A) -> bool) -> Option<uint> {
        iter::position(self, f)
    }
}

impl<A: Eq> &[A]: iter::EqIter<A> {
    pure fn contains(x: &A) -> bool { iter::contains(self, x) }
    pure fn count(x: &A) -> uint { iter::count(self, x) }
}

impl<A: Copy> &[A]: iter::CopyableIter<A> {
    pure fn filter_to_vec(pred: fn(A) -> bool) -> ~[A] {
        iter::filter_to_vec(self, pred)
    }
    pure fn map_to_vec<B>(op: fn(v: &A) -> B) -> ~[B] {
        iter::map_to_vec(self, op)
    }
    pure fn to_vec() -> ~[A] { iter::to_vec(self) }

    // FIXME--bug in resolve prevents this from working (#2611)
    // fn flat_map_to_vec<B:copy,IB:base_iter<B>>(op: fn(A) -> IB) -> ~[B] {
    //     iter::flat_map_to_vec(self, op)
    // }

    pure fn find(p: fn(A) -> bool) -> Option<A> { iter::find(self, p) }
}

impl<A: Copy Ord> &[A]: iter::CopyableOrderedIter<A> {
    pure fn min() -> A { iter::min(self) }
    pure fn max() -> A { iter::max(self) }
}
// ___________________________________________________________________________

#[cfg(test)]
mod tests {
    #[legacy_exports];

    fn square(n: uint) -> uint { return n * n; }

    fn square_ref(n: &uint) -> uint { return square(*n); }

    pure fn is_three(&&n: uint) -> bool { return n == 3u; }

    pure fn is_odd(&&n: uint) -> bool { return n % 2u == 1u; }

    pure fn is_equal(&&x: uint, &&y:uint) -> bool { return x == y; }

    fn square_if_odd(&&n: uint) -> Option<uint> {
        return if n % 2u == 1u { Some(n * n) } else { None };
    }

    fn add(&&x: uint, &&y: uint) -> uint { return x + y; }

    #[test]
    fn test_unsafe_ptrs() {
        unsafe {
            // Test on-stack copy-from-buf.
            let a = ~[1, 2, 3];
            let mut ptr = raw::to_ptr(a);
            let b = raw::from_buf(ptr, 3u);
            assert (len(b) == 3u);
            assert (b[0] == 1);
            assert (b[1] == 2);
            assert (b[2] == 3);

            // Test on-heap copy-from-buf.
            let c = ~[1, 2, 3, 4, 5];
            ptr = raw::to_ptr(c);
            let d = raw::from_buf(ptr, 5u);
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
        assert (n.is_none());
        n = last_opt(~[1, 2, 3]);
        assert (n == Some(3));
        n = last_opt(~[1, 2, 3, 4, 5]);
        assert (n == Some(5));
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
    fn test_swap_remove() {
        let mut v = ~[1, 2, 3, 4, 5];
        let mut e = swap_remove(v, 0);
        assert (len(v) == 4);
        assert e == 1;
        assert (v[0] == 5);
        e = swap_remove(v, 3);
        assert (len(v) == 3);
        assert e == 4;
        assert (v[0] == 5);
        assert (v[1] == 2);
        assert (v[2] == 3);
    }

    #[test]
    fn test_swap_remove_noncopyable() {
        // Tests that we don't accidentally run destructors twice.
        let mut v = ~[::private::exclusive(()), ::private::exclusive(()),
                      ::private::exclusive(())];
        let mut _e = swap_remove(v, 0);
        assert (len(v) == 2);
        _e = swap_remove(v, 1);
        assert (len(v) == 1);
        _e = swap_remove(v, 0);
        assert (len(v) == 0);
    }

    #[test]
    fn test_push() {
        // Test on-stack push().
        let mut v = ~[];
        v.push(1);
        assert (len(v) == 1u);
        assert (v[0] == 1);

        // Test on-heap push().
        v.push(2);
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
        let mut v = ~[1, 2, 3];
        grow_set(v, 4u, 4, 5);
        assert (len(v) == 5u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);
        assert (v[3] == 4);
        assert (v[4] == 5);
    }

    #[test]
    fn test_truncate() {
        let mut v = ~[@6,@5,@4];
        truncate(v, 1);
        assert(v.len() == 1);
        assert(*(v[0]) == 6);
        // If the unsafe block didn't drop things properly, we blow up here.
    }

    #[test]
    fn test_dedup() {
        fn case(-a: ~[uint], -b: ~[uint]) {
            let mut v = a;
            dedup(v);
            assert(v == b);
        }
        case(~[], ~[]);
        case(~[1], ~[1]);
        case(~[1,1], ~[1]);
        case(~[1,2,3], ~[1,2,3]);
        case(~[1,1,2,3], ~[1,2,3]);
        case(~[1,2,2,3], ~[1,2,3]);
        case(~[1,2,3,3], ~[1,2,3]);
        case(~[1,1,2,2,2,3,3], ~[1,2,3]);
    }

    #[test]
    fn test_dedup_unique() {
        let mut v0 = ~[~1, ~1, ~2, ~3];
        dedup(v0);
        let mut v1 = ~[~1, ~2, ~2, ~3];
        dedup(v1);
        let mut v2 = ~[~1, ~2, ~3, ~3];
        dedup(v2);
        /*
         * If the ~pointers were leaked or otherwise misused, valgrind and/or
         * rustrt should raise errors.
         */
    }

    #[test]
    fn test_dedup_shared() {
        let mut v0 = ~[@1, @1, @2, @3];
        dedup(v0);
        let mut v1 = ~[@1, @2, @2, @3];
        dedup(v1);
        let mut v2 = ~[@1, @2, @3, @3];
        dedup(v2);
        /*
         * If the @pointers were leaked or otherwise misused, valgrind and/or
         * rustrt should raise errors.
         */
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
        fn times(&&x: int, &&y: int) -> int { return x * y; }
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

        fn halve(&&i: int) -> Option<int> {
            if i % 2 == 0 {
                return option::Some::<int>(i / 2);
            } else { return option::None::<int>; }
        }
        fn halve_for_sure(i: &int) -> int { return *i / 2; }
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
    fn test_each_empty() {
        for each::<int>(~[]) |_v| {
            fail; // should never be executed
        }
    }

    #[test]
    fn test_iter_nonempty() {
        let mut i = 0;
        for each(~[1, 2, 3]) |v| {
            i += *v;
        }
        assert i == 6;
    }

    #[test]
    fn test_iteri() {
        let mut i = 0;
        for eachi(~[1, 2, 3]) |j, v| {
            if i == 0 { assert *v == 1; }
            assert j + 1u == *v as uint;
            i += *v;
        }
        assert i == 6;
    }

    #[test]
    fn test_reach_empty() {
        for rev_each::<int>(~[]) |_v| {
            fail; // should never execute
        }
    }

    #[test]
    fn test_reach_nonempty() {
        let mut i = 0;
        for rev_each(~[1, 2, 3]) |v| {
            if i == 0 { assert *v == 3; }
            i += *v
        }
        assert i == 6;
    }

    #[test]
    fn test_reachi() {
        let mut i = 0;
        for rev_eachi(~[0, 1, 2]) |j, v| {
            if i == 0 { assert *v == 2; }
            assert j == *v as uint;
            i += *v;
        }
        assert i == 3;
    }

    #[test]
    fn test_permute() {
        let mut results: ~[~[int]];

        results = ~[];
        permute(~[], |v| results.push(copy v));
        assert results == ~[~[]];

        results = ~[];
        permute(~[7], |v| results.push(copy v));
        assert results == ~[~[7]];

        results = ~[];
        permute(~[1,1], |v| results.push(copy v));
        assert results == ~[~[1,1],~[1,1]];

        results = ~[];
        permute(~[5,2,0], |v| results.push(copy v));
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
        assert position_elem(~[], 1).is_none();

        let v1 = ~[1, 2, 3, 3, 2, 5];
        assert position_elem(v1, 1) == Some(0u);
        assert position_elem(v1, 2) == Some(1u);
        assert position_elem(v1, 5) == Some(5u);
        assert position_elem(v1, 4).is_none();
    }

    #[test]
    fn test_position() {
        fn less_than_three(&&i: int) -> bool { return i < 3; }
        fn is_eighteen(&&i: int) -> bool { return i == 18; }

        assert position(~[], less_than_three).is_none();

        let v1 = ~[5, 4, 3, 2, 1];
        assert position(v1, less_than_three) == Some(3u);
        assert position(v1, is_eighteen).is_none();
    }

    #[test]
    fn test_position_between() {
        assert position_between(~[], 0u, 0u, f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position_between(v, 0u, 0u, f).is_none();
        assert position_between(v, 0u, 1u, f).is_none();
        assert position_between(v, 0u, 2u, f) == Some(1u);
        assert position_between(v, 0u, 3u, f) == Some(1u);
        assert position_between(v, 0u, 4u, f) == Some(1u);

        assert position_between(v, 1u, 1u, f).is_none();
        assert position_between(v, 1u, 2u, f) == Some(1u);
        assert position_between(v, 1u, 3u, f) == Some(1u);
        assert position_between(v, 1u, 4u, f) == Some(1u);

        assert position_between(v, 2u, 2u, f).is_none();
        assert position_between(v, 2u, 3u, f).is_none();
        assert position_between(v, 2u, 4u, f) == Some(3u);

        assert position_between(v, 3u, 3u, f).is_none();
        assert position_between(v, 3u, 4u, f) == Some(3u);

        assert position_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn test_find() {
        assert find(~[], f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find(v, f) == Some((1, 'b'));
        assert find(v, g).is_none();
    }

    #[test]
    fn test_find_between() {
        assert find_between(~[], 0u, 0u, f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find_between(v, 0u, 0u, f).is_none();
        assert find_between(v, 0u, 1u, f).is_none();
        assert find_between(v, 0u, 2u, f) == Some((1, 'b'));
        assert find_between(v, 0u, 3u, f) == Some((1, 'b'));
        assert find_between(v, 0u, 4u, f) == Some((1, 'b'));

        assert find_between(v, 1u, 1u, f).is_none();
        assert find_between(v, 1u, 2u, f) == Some((1, 'b'));
        assert find_between(v, 1u, 3u, f) == Some((1, 'b'));
        assert find_between(v, 1u, 4u, f) == Some((1, 'b'));

        assert find_between(v, 2u, 2u, f).is_none();
        assert find_between(v, 2u, 3u, f).is_none();
        assert find_between(v, 2u, 4u, f) == Some((3, 'b'));

        assert find_between(v, 3u, 3u, f).is_none();
        assert find_between(v, 3u, 4u, f) == Some((3, 'b'));

        assert find_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn test_rposition() {
        assert find(~[], f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position(v, f) == Some(1u);
        assert position(v, g).is_none();
    }

    #[test]
    fn test_rposition_between() {
        assert rposition_between(~[], 0u, 0u, f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rposition_between(v, 0u, 0u, f).is_none();
        assert rposition_between(v, 0u, 1u, f).is_none();
        assert rposition_between(v, 0u, 2u, f) == Some(1u);
        assert rposition_between(v, 0u, 3u, f) == Some(1u);
        assert rposition_between(v, 0u, 4u, f) == Some(3u);

        assert rposition_between(v, 1u, 1u, f).is_none();
        assert rposition_between(v, 1u, 2u, f) == Some(1u);
        assert rposition_between(v, 1u, 3u, f) == Some(1u);
        assert rposition_between(v, 1u, 4u, f) == Some(3u);

        assert rposition_between(v, 2u, 2u, f).is_none();
        assert rposition_between(v, 2u, 3u, f).is_none();
        assert rposition_between(v, 2u, 4u, f) == Some(3u);

        assert rposition_between(v, 3u, 3u, f).is_none();
        assert rposition_between(v, 3u, 4u, f) == Some(3u);

        assert rposition_between(v, 4u, 4u, f).is_none();
    }

    #[test]
    fn test_rfind() {
        assert rfind(~[], f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind(v, f) == Some((3, 'b'));
        assert rfind(v, g).is_none();
    }

    #[test]
    fn test_rfind_between() {
        assert rfind_between(~[], 0u, 0u, f).is_none();

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let mut v = ~[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind_between(v, 0u, 0u, f).is_none();
        assert rfind_between(v, 0u, 1u, f).is_none();
        assert rfind_between(v, 0u, 2u, f) == Some((1, 'b'));
        assert rfind_between(v, 0u, 3u, f) == Some((1, 'b'));
        assert rfind_between(v, 0u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 1u, 1u, f).is_none();
        assert rfind_between(v, 1u, 2u, f) == Some((1, 'b'));
        assert rfind_between(v, 1u, 3u, f) == Some((1, 'b'));
        assert rfind_between(v, 1u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 2u, 2u, f).is_none();
        assert rfind_between(v, 2u, 3u, f).is_none();
        assert rfind_between(v, 2u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 3u, 3u, f).is_none();
        assert rfind_between(v, 3u, 4u, f) == Some((3, 'b'));

        assert rfind_between(v, 4u, 4u, f).is_none();
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
            let addr = raw::to_ptr(x);
            let x_mut = to_mut(x);
            let addr_mut = raw::to_ptr(x_mut);
            assert addr == addr_mut;
        }
    }

    #[test]
    fn from_mut_no_copy() {
        unsafe {
            let x = ~[mut 1, 2, 3];
            let addr = raw::to_ptr(x);
            let x_imm = from_mut(x);
            let addr_imm = raw::to_ptr(x_imm);
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
        reserve(&mut v, 10u);
        assert capacity(v) == 10u;
        let mut v = ~[0u32];
        reserve(&mut v, 10u);
        assert capacity(v) == 10u;
    }

/*
    #[test]
    #[ignore] // region inference doesn't work well enough for this yet.
    fn test_view() {
        let v = ~[1, 2, 3, 4, 5];
        let v = view(v, 1u, 3u);
        assert(len(v) == 2u);
        assert(v[0] == 2);
        assert(v[1] == 3);
    }
*/
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
