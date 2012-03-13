import option::{some, none};
import uint::next_power_of_two;
import ptr::addr_of;

export init_op;
export is_empty;
export is_not_empty;
export same_length;
export reserve;
export len;
export init_fn;
export init_elt;
export to_mut;
export from_mut;
export head;
export tail;
export tailn;
export init;
export last;
export last_opt;
export slice;
export split;
export splitn;
export rsplit;
export rsplitn;
export shift;
export pop;
export push;
export grow;
export grow_fn;
export grow_set;
export map;
export map2;
export filter_map;
export filter;
export concat;
export connect;
export foldl;
export foldr;
export any;
export any2;
export all;
export all2;
export contains;
export count;
export find;
export find_from;
export rfind;
export rfind_from;
export position_elt;
export position;
export position_from;
export position_elt;
export rposition;
export rposition_from;
export unzip;
export zip;
export swap;
export reverse;
export reversed;
export iter;
export iter2;
export iteri;
export riter;
export riteri;
export permute;
export windowed;
export as_buf;
export as_mut_buf;
export vec_len;
export unsafe;
export u8;

#[abi = "rust-intrinsic"]
native mod rusti {
    fn vec_len<T>(&&v: [const T]) -> libc::size_t;
}

#[abi = "cdecl"]
native mod rustrt {
    fn vec_reserve_shared<T>(t: *sys::type_desc,
                             &v: [const T],
                             n: libc::size_t);
    fn vec_from_buf_shared<T>(t: *sys::type_desc,
                              ptr: *T,
                              count: libc::size_t) -> [T];
}

#[doc = "A function used to initialize the elements of a vector"]
type init_op<T> = fn(uint) -> T;

#[doc = "Returns true if a vector contains no elements"]
pure fn is_empty<T>(v: [const T]) -> bool {
    // FIXME: This would be easier if we could just call len
    for t: T in v { ret false; }
    ret true;
}

#[doc = "Returns true if a vector contains some elements"]
pure fn is_not_empty<T>(v: [const T]) -> bool { ret !is_empty(v); }

#[doc = "Returns true if two vectors have the same length"]
pure fn same_length<T, U>(xs: [const T], ys: [const U]) -> bool {
    vec::len(xs) == vec::len(ys)
}

#[doc = "
Reserves capacity for `n` elements in the given vector.

If the capacity for `v` is already equal to or greater than the requested
capacity, then no action is taken.

# Arguments

* v - A vector
* n - The number of elements to reserve space for
"]
fn reserve<T>(&v: [const T], n: uint) {
    rustrt::vec_reserve_shared(sys::get_type_desc::<T>(), v, n);
}

#[doc = "Returns the length of a vector"]
#[inline(always)]
pure fn len<T>(v: [const T]) -> uint { unchecked { rusti::vec_len(v) } }

#[doc = "
Creates and initializes an immutable vector.

Creates an immutable vector of size `n_elts` and initializes the elements
to the value returned by the function `op`.
"]
fn init_fn<T>(n_elts: uint, op: init_op<T>) -> [T] {
    let mut v = [];
    reserve(v, n_elts);
    let mut i: uint = 0u;
    while i < n_elts { v += [op(i)]; i += 1u; }
    ret v;
}

#[doc = "
Creates and initializes an immutable vector.

Creates an immutable vector of size `n_elts` and initializes the elements
to the value `t`.
"]
fn init_elt<T: copy>(n_elts: uint, t: T) -> [T] {
    let mut v = [];
    reserve(v, n_elts);
    let mut i: uint = 0u;
    while i < n_elts { v += [t]; i += 1u; }
    ret v;
}

// FIXME: Possible typestate postcondition:
// len(result) == len(v) (needs issue #586)
#[doc = "Produces a mutable vector from an immutable vector."]
fn to_mut<T>(+v: [T]) -> [mutable T] unsafe {
    let r = ::unsafe::reinterpret_cast(v);
    ::unsafe::leak(v);
    r
}

#[doc = "Produces an immutable vector from a mutable vector."]
fn from_mut<T>(+v: [mutable T]) -> [T] unsafe {
    let r = ::unsafe::reinterpret_cast(v);
    ::unsafe::leak(v);
    r
}

// Accessors

#[doc = "Returns the first element of a vector"]
pure fn head<T: copy>(v: [const T]) -> T { v[0] }

#[doc = "Returns all but the first element of a vector"]
fn tail<T: copy>(v: [const T]) -> [T] {
    ret slice(v, 1u, len(v));
}

#[doc = "Returns all but the first `n` elements of a vector"]
fn tailn<T: copy>(v: [const T], n: uint) -> [T] {
    slice(v, n, len(v))
}

// FIXME: This name is sort of confusing next to init_fn, etc
// but this is the name haskell uses for this function,
// along with head/tail/last.
#[doc = "Returns all but the last elemnt of a vector"]
fn init<T: copy>(v: [const T]) -> [T] {
    assert len(v) != 0u;
    slice(v, 0u, len(v) - 1u)
}

#[doc = "
Returns the last element of a `v`, failing if the vector is empty.
"]
pure fn last<T: copy>(v: [const T]) -> T {
    if len(v) == 0u { fail "last_unsafe: empty vector" }
    v[len(v) - 1u]
}

#[doc = "
Returns some(x) where `x` is the last element of a vector `v`,
or none if the vector is empty.
"]
pure fn last_opt<T: copy>(v: [const T]) -> option<T> {
   if len(v) == 0u { ret none; }
    some(v[len(v) - 1u])
}

#[doc = "Returns a copy of the elements from [`start`..`end`) from `v`."]
fn slice<T: copy>(v: [const T], start: uint, end: uint) -> [T] {
    assert (start <= end);
    assert (end <= len(v));
    let mut result = [];
    reserve(result, end - start);
    let mut i = start;
    while i < end { result += [v[i]]; i += 1u; }
    ret result;
}

#[doc = "
Split the vector `v` by applying each element against the predicate `f`.
"]
fn split<T: copy>(v: [const T], f: fn(T) -> bool) -> [[T]] {
    let ln = len(v);
    if (ln == 0u) { ret [] }

    let mut start = 0u;
    let mut result = [];
    while start < ln {
        alt position_from(v, start, ln, f) {
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

#[doc = "
Split the vector `v` by applying each element against the predicate `f` up
to `n` times.
"]
fn splitn<T: copy>(v: [const T], n: uint, f: fn(T) -> bool) -> [[T]] {
    let ln = len(v);
    if (ln == 0u) { ret [] }

    let mut start = 0u;
    let mut count = n;
    let mut result = [];
    while start < ln && count > 0u {
        alt position_from(v, start, ln, f) {
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

#[doc = "
Reverse split the vector `v` by applying each element against the predicate
`f`.
"]
fn rsplit<T: copy>(v: [const T], f: fn(T) -> bool) -> [[T]] {
    let ln = len(v);
    if (ln == 0u) { ret [] }

    let mut end = ln;
    let mut result = [];
    while end > 0u {
        alt rposition_from(v, 0u, end, f) {
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

#[doc = "
Reverse split the vector `v` by applying each element against the predicate
`f` up to `n times.
"]
fn rsplitn<T: copy>(v: [const T], n: uint, f: fn(T) -> bool) -> [[T]] {
    let ln = len(v);
    if (ln == 0u) { ret [] }

    let mut end = ln;
    let mut count = n;
    let mut result = [];
    while end > 0u && count > 0u {
        alt rposition_from(v, 0u, end, f) {
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

#[doc = "Removes the first element from a vector and return it"]
fn shift<T: copy>(&v: [const T]) -> T {
    let ln = len::<T>(v);
    assert (ln > 0u);
    let e = v[0];
    v = slice::<T>(v, 1u, ln);
    ret e;
}

#[doc = "Remove the last element from a vector and return it"]
fn pop<T>(&v: [const T]) -> T unsafe {
    let ln = len(v);
    assert ln > 0u;
    let valptr = ptr::mut_addr_of(v[ln - 1u]);
    let val <- *valptr;
    unsafe::set_len(v, ln - 1u);
    val
}

#[doc = "Append an element to a vector"]
fn push<T: copy>(&v: [const T], initval: T) {
    v += [initval];
}


// Appending

#[doc = "
Expands a vector in place, initializing the new elements to a given value

# Arguments

* v - The vector to grow
* n - The number of elements to add
* initval - The value for the new elements
"]
fn grow<T: copy>(&v: [const T], n: uint, initval: T) {
    reserve(v, next_power_of_two(len(v) + n));
    let mut i: uint = 0u;
    while i < n { v += [initval]; i += 1u; }
}

#[doc = "
Expands a vector in place, initializing the new elements to the result of
a function

Function `init_op` is called `n` times with the values [0..`n`)

# Arguments

* v - The vector to grow
* n - The number of elements to add
* init_op - A function to call to retreive each appended element's
            value
"]
fn grow_fn<T>(&v: [const T], n: uint, op: init_op<T>) {
    reserve(v, next_power_of_two(len(v) + n));
    let mut i: uint = 0u;
    while i < n { v += [op(i)]; i += 1u; }
}

#[doc = "
Sets the value of a vector element at a given index, growing the vector as
needed

Sets the element at position `index` to `val`. If `index` is past the end
of the vector, expands the vector by replicating `initval` to fill the
intervening space.
"]
fn grow_set<T: copy>(&v: [mutable T], index: uint, initval: T, val: T) {
    if index >= len(v) { grow(v, index - len(v) + 1u, initval); }
    v[index] = val;
}


// Functional utilities

#[doc ="
Apply a function to each element of a vector and return the results
"]
fn map<T, U>(v: [T], f: fn(T) -> U) -> [U] {
    let mut result = [];
    reserve(result, len(v));
    for elem: T in v { result += [f(elem)]; }
    ret result;
}

#[doc = "
Apply a function to each pair of elements and return the results
"]
fn map2<T: copy, U: copy, V>(v0: [const T], v1: [const U],
                             f: fn(T, U) -> V) -> [V] {
    let v0_len = len(v0);
    if v0_len != len(v1) { fail; }
    let mut u: [V] = [];
    let mut i = 0u;
    while i < v0_len { u += [f(copy v0[i], copy v1[i])]; i += 1u; }
    ret u;
}

#[doc = "
Apply a function to each element of a vector and return the results

If function `f` returns `none` then that element is excluded from
the resulting vector.
"]
fn filter_map<T: copy, U: copy>(v: [const T], f: fn(T) -> option<U>)
    -> [U] {
    let mut result = [];
    for elem: T in v {
        alt f(copy elem) {
          none {/* no-op */ }
          some(result_elem) { result += [result_elem]; }
        }
    }
    ret result;
}

#[doc = "
Construct a new vector from the elements of a vector for which some predicate
holds.

Apply function `f` to each element of `v` and return a vector containing
only those elements for which `f` returned true.
"]
fn filter<T: copy>(v: [T], f: fn(T) -> bool) -> [T] {
    let mut result = [];
    for elem: T in v {
        if f(elem) { result += [elem]; }
    }
    ret result;
}

#[doc = "
Concatenate a vector of vectors.

Flattens a vector of vectors of T into a single vector of T.
"]
fn concat<T: copy>(v: [const [const T]]) -> [T] {
    let mut new: [T] = [];
    for inner: [T] in v { new += inner; }
    ret new;
}

#[doc = "
Concatenate a vector of vectors, placing a given separator between each
"]
fn connect<T: copy>(v: [const [const T]], sep: T) -> [T] {
    let mut new: [T] = [];
    let mut first = true;
    for inner: [T] in v {
        if first { first = false; } else { push(new, sep); }
        new += inner;
    }
    ret new;
}

#[doc = "Reduce a vector from left to right"]
fn foldl<T: copy, U>(z: T, v: [const U], p: fn(T, U) -> T) -> T {
    let mut accum = z;
    iter(v) { |elt|
        accum = p(accum, elt);
    }
    ret accum;
}

#[doc = "Reduce a vector from right to left"]
fn foldr<T, U: copy>(v: [const T], z: U, p: fn(T, U) -> U) -> U {
    let mut accum = z;
    riter(v) { |elt|
        accum = p(elt, accum);
    }
    ret accum;
}

#[doc = "
Return true if a predicate matches any elements

If the vector contains no elements then false is returned.
"]
fn any<T>(v: [T], f: fn(T) -> bool) -> bool {
    for elem: T in v { if f(elem) { ret true; } }
    ret false;
}

#[doc = "
Return true if a predicate matches any elements in both vectors.

If the vectors contains no elements then false is returned.
"]
fn any2<T, U>(v0: [const T], v1: [U], f: fn(T, U) -> bool) -> bool {
    let v0_len = len(v0);
    let v1_len = len(v1);
    let mut i = 0u;
    while i < v0_len && i < v1_len {
        if f(v0[i], v1[i]) { ret true; };
        i += 1u;
    }
    ret false;
}

#[doc = "
Return true if a predicate matches all elements

If the vector contains no elements then true is returned.
"]
fn all<T>(v: [T], f: fn(T) -> bool) -> bool {
    for elem: T in v { if !f(elem) { ret false; } }
    ret true;
}

#[doc = "
Return true if a predicate matches all elements in both vectors.

If the vectors are not the same size then false is returned.
"]
fn all2<T, U>(v0: [const T], v1: [const U], f: fn(T, U) -> bool) -> bool {
    let v0_len = len(v0);
    if v0_len != len(v1) { ret false; }
    let mut i = 0u;
    while i < v0_len { if !f(v0[i], v1[i]) { ret false; }; i += 1u; }
    ret true;
}

#[doc = "Return true if a vector contains an element with the given value"]
fn contains<T>(v: [const T], x: T) -> bool {
    for elt: T in v { if x == elt { ret true; } }
    ret false;
}

#[doc = "Returns the number of elements that are equal to a given value"]
fn count<T>(v: [const T], x: T) -> uint {
    let mut cnt = 0u;
    for elt: T in v { if x == elt { cnt += 1u; } }
    ret cnt;
}

#[doc = "
Search for the first element that matches a given predicate

Apply function `f` to each element of `v`, starting from the first.
When function `f` returns true then an option containing the element
is returned. If `f` matches no elements then none is returned.
"]
fn find<T: copy>(v: [const T], f: fn(T) -> bool) -> option<T> {
    find_from(v, 0u, len(v), f)
}

#[doc = "
Search for the first element that matches a given predicate within a range

Apply function `f` to each element of `v` within the range [`start`, `end`).
When function `f` returns true then an option containing the element
is returned. If `f` matches no elements then none is returned.
"]
fn find_from<T: copy>(v: [const T], start: uint, end: uint,
                      f: fn(T) -> bool) -> option<T> {
    option::map(position_from(v, start, end, f)) { |i| v[i] }
}

#[doc = "
Search for the last element that matches a given predicate

Apply function `f` to each element of `v` in reverse order. When function `f`
returns true then an option containing the element is returned. If `f`
matches no elements then none is returned.
"]
fn rfind<T: copy>(v: [const T], f: fn(T) -> bool) -> option<T> {
    rfind_from(v, 0u, len(v), f)
}

#[doc = "
Search for the last element that matches a given predicate within a range

Apply function `f` to each element of `v` in reverse order within the range
[`start`, `end`). When function `f` returns true then an option containing
the element is returned. If `f` matches no elements then none is returned.
"]
fn rfind_from<T: copy>(v: [const T], start: uint, end: uint,
                       f: fn(T) -> bool) -> option<T> {
    option::map(rposition_from(v, start, end, f)) { |i| v[i] }
}

#[doc = "Find the first index containing a matching value"]
fn position_elt<T>(v: [const T], x: T) -> option<uint> {
    position(v) { |y| x == y }
}

#[doc = "
Find the first index matching some predicate

Apply function `f` to each element of `v`.  When function `f` returns true
then an option containing the index is returned. If `f` matches no elements
then none is returned.
"]
fn position<T>(v: [const T], f: fn(T) -> bool) -> option<uint> {
    position_from(v, 0u, len(v), f)
}

#[doc = "
Find the first index matching some predicate within a range

Apply function `f` to each element of `v` between the range [`start`, `end`).
When function `f` returns true then an option containing the index is
returned. If `f` matches no elements then none is returned.
"]
fn position_from<T>(v: [const T], start: uint, end: uint,
                    f: fn(T) -> bool) -> option<uint> {
    assert start <= end;
    assert end <= len(v);
    let mut i = start;
    while i < end { if f(v[i]) { ret some::<uint>(i); } i += 1u; }
    ret none;
}

#[doc = "Find the last index containing a matching value"]
fn rposition_elt<T>(v: [const T], x: T) -> option<uint> {
    rposition(v) { |y| x == y }
}

#[doc = "
Find the last index matching some predicate

Apply function `f` to each element of `v` in reverse order.  When function
`f` returns true then an option containing the index is returned. If `f`
matches no elements then none is returned.
"]
fn rposition<T>(v: [const T], f: fn(T) -> bool) -> option<uint> {
    rposition_from(v, 0u, len(v), f)
}

#[doc = "
Find the last index matching some predicate within a range

Apply function `f` to each element of `v` in reverse order between the range
[`start`, `end`). When function `f` returns true then an option containing
the index is returned. If `f` matches no elements then none is returned.
"]
fn rposition_from<T>(v: [const T], start: uint, end: uint,
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
#[doc = "
Convert a vector of pairs into a pair of vectors

Returns a tuple containing two vectors where the i-th element of the first
vector contains the first element of the i-th tuple of the input vector,
and the i-th element of the second vector contains the second element
of the i-th tuple of the input vector.
"]
fn unzip<T: copy, U: copy>(v: [const (T, U)]) -> ([T], [U]) {
    let mut as = [], bs = [];
    for (a, b) in v { as += [a]; bs += [b]; }
    ret (as, bs);
}

#[doc = "
Convert two vectors to a vector of pairs

Returns a vector of tuples, where the i-th tuple contains contains the
i-th elements from each of the input vectors.
"]
fn zip<T: copy, U: copy>(v: [const T], u: [const U]) -> [(T, U)] {
    let mut zipped = [];
    let sz = len(v);
    let mut i = 0u;
    assert sz == len(u);
    while i < sz { zipped += [(v[i], u[i])]; i += 1u; }
    ret zipped;
}

#[doc = "
Swaps two elements in a vector

# Arguments

* v  The input vector
* a - The index of the first element
* b - The index of the second element
"]
fn swap<T>(v: [mutable T], a: uint, b: uint) {
    v[a] <-> v[b];
}

#[doc = "Reverse the order of elements in a vector, in place"]
fn reverse<T>(v: [mutable T]) {
    let mut i: uint = 0u;
    let ln = len::<T>(v);
    while i < ln / 2u { v[i] <-> v[ln - i - 1u]; i += 1u; }
}


#[doc = "Returns a vector with the order of elements reversed"]
fn reversed<T: copy>(v: [const T]) -> [T] {
    let mut rs: [T] = [];
    let mut i = len::<T>(v);
    if i == 0u { ret rs; } else { i -= 1u; }
    while i != 0u { rs += [v[i]]; i -= 1u; }
    rs += [v[0]];
    ret rs;
}

#[doc = "
Iterates over a vector

Iterates over vector `v` and, for each element, calls function `f` with the
element's value.
"]
#[inline(always)]
fn iter<T>(v: [const T], f: fn(T)) {
    unsafe {
        let mut n = vec::len(v);
        let mut p = unsafe::to_ptr(v);
        while n > 0u {
            f(*p);
            p = ptr::offset(p, 1u);
            n -= 1u;
        }
    }
}

#[doc = "Iterates over two vectors in parallel"]
#[inline]
fn iter2<U, T>(v: [ U], v2: [const T], f: fn(U, T)) {
    let mut i = 0;
    for elt in v { f(elt, v2[i]); i += 1; }
}

#[doc = "
Iterates over a vector's elements and indexes

Iterates over vector `v` and, for each element, calls function `f` with the
element's value and index.
"]
#[inline(always)]
fn iteri<T>(v: [const T], f: fn(uint, T)) {
    let mut i = 0u;
    let l = len(v);
    while i < l { f(i, v[i]); i += 1u; }
}

#[doc = "
Iterates over a vector in reverse

Iterates over vector `v` and, for each element, calls function `f` with the
element's value.
"]
fn riter<T>(v: [const T], f: fn(T)) {
    riteri(v) { |_i, v| f(v) }
}

#[doc ="
Iterates over a vector's elements and indexes in reverse

Iterates over vector `v` and, for each element, calls function `f` with the
element's value and index.
"]
fn riteri<T>(v: [const T], f: fn(uint, T)) {
    let mut i = len(v);
    while 0u < i {
        i -= 1u;
        f(i, v[i]);
    };
}

#[doc = "
Iterate over all permutations of vector `v`.

Permutations are produced in lexicographic order with respect to the order of
elements in `v` (so if `v` is sorted then the permutations are
lexicographically sorted).

The total number of permutations produced is `len(v)!`.  If `v` contains
repeated elements, then some permutations are repeated.
"]
fn permute<T: copy>(v: [T], put: fn([T])) {
  let ln = len(v);
  if ln == 0u {
    put([]);
  } else {
    let mut i = 0u;
    while i < ln {
      let elt = v[i];
      let rest = slice(v, 0u, i) + slice(v, i+1u, ln);
      permute(rest) {|permutation| put([elt] + permutation)}
      i += 1u;
    }
  }
}

fn windowed <TT: copy> (nn: uint, xx: [const TT]) -> [[TT]] {
   let mut ww = [];

   assert 1u <= nn;

   vec::iteri (xx, {|ii, _x|
      let len = vec::len(xx);

      if ii+nn <= len {
         let w = vec::slice ( xx, ii, ii+nn );
         vec::push (ww, w);
      }
   });

   ret ww;
}

#[doc = "
Work with the buffer of a vector.

Allows for unsafe manipulation of vector contents, which is useful for native
interop.
"]
fn as_buf<E,T>(v: [const E], f: fn(*E) -> T) -> T unsafe {
    let buf = unsafe::to_ptr(v); f(buf)
}

fn as_mut_buf<E,T>(v: [mutable E], f: fn(*mutable E) -> T) -> T unsafe {
    let buf = unsafe::to_ptr(v) as *mutable E; f(buf)
}

impl vec_len<T> for [T] {
    #[inline(always)]
    fn len() -> uint { len(self) }
}

mod unsafe {
    // FIXME: This should have crate visibility
    type vec_repr = {mutable fill: uint, mutable alloc: uint, data: u8};

    #[doc = "
    Constructs a vector from an unsafe pointer to a buffer

    # Arguments

    * ptr - An unsafe pointer to a buffer of `T`
    * elts - The number of elements in the buffer
    "]
    #[inline(always)]
    unsafe fn from_buf<T>(ptr: *T, elts: uint) -> [T] {
        ret rustrt::vec_from_buf_shared(sys::get_type_desc::<T>(),
                                        ptr, elts);
    }

    #[doc = "
    Sets the length of a vector

    This well explicitly set the size of the vector, without actually
    modifing its buffers, so it is up to the caller to ensure that
    the vector is actually the specified size.
    "]
    #[inline(always)]
    unsafe fn set_len<T>(&v: [const T], new_len: uint) {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).fill = new_len * sys::size_of::<T>();
    }

    #[doc = "
    Returns an unsafe pointer to the vector's buffer

    The caller must ensure that the vector outlives the pointer this
    function returns, or else it will end up pointing to garbage.

    Modifying the vector may cause its buffer to be reallocated, which
    would also make any pointers to it invalid.
    "]
    #[inline(always)]
    unsafe fn to_ptr<T>(v: [const T]) -> *T {
        let repr: **vec_repr = ::unsafe::reinterpret_cast(addr_of(v));
        ret ::unsafe::reinterpret_cast(addr_of((**repr).data));
    }
}

mod u8 {
    export cmp;
    export lt, le, eq, ne, ge, gt;
    export hash;

    #[doc = "Bytewise string comparison"]
    pure fn cmp(&&a: [u8], &&b: [u8]) -> int unsafe {
        let a_len = len(a);
        let b_len = len(b);
        let n = uint::min(a_len, b_len) as libc::size_t;
        let r = libc::memcmp(unsafe::to_ptr(a) as *libc::c_void,
                             unsafe::to_ptr(b) as *libc::c_void, n) as int;

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

    #[doc = "Bytewise less than or equal"]
    pure fn lt(&&a: [u8], &&b: [u8]) -> bool { cmp(a, b) < 0 }

    #[doc = "Bytewise less than or equal"]
    pure fn le(&&a: [u8], &&b: [u8]) -> bool { cmp(a, b) <= 0 }

    #[doc = "Bytewise equality"]
    pure fn eq(&&a: [u8], &&b: [u8]) -> bool unsafe { cmp(a, b) == 0 }

    #[doc = "Bytewise inequality"]
    pure fn ne(&&a: [u8], &&b: [u8]) -> bool unsafe { cmp(a, b) != 0 }

    #[doc ="Bytewise greater than or equal"]
    pure fn ge(&&a: [u8], &&b: [u8]) -> bool { cmp(a, b) >= 0 }

    #[doc = "Bytewise greater than"]
    pure fn gt(&&a: [u8], &&b: [u8]) -> bool { cmp(a, b) > 0 }

    #[doc = "String hash function"]
    fn hash(&&s: [u8]) -> uint {
        // djb hash.
        // FIXME: replace with murmur.

        let mut u: uint = 5381u;
        vec::iter(s, { |c| u *= 33u; u += c as uint; });
        ret u;
    }
}

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
    fn test_unsafe_ptrs() unsafe {
        // Test on-stack copy-from-buf.
        let a = [1, 2, 3];
        let ptr = unsafe::to_ptr(a);
        let b = unsafe::from_buf(ptr, 3u);
        assert (len(b) == 3u);
        assert (b[0] == 1);
        assert (b[1] == 2);
        assert (b[2] == 3);

        // Test on-heap copy-from-buf.
        let c = [1, 2, 3, 4, 5];
        ptr = unsafe::to_ptr(c);
        let d = unsafe::from_buf(ptr, 5u);
        assert (len(d) == 5u);
        assert (d[0] == 1);
        assert (d[1] == 2);
        assert (d[2] == 3);
        assert (d[3] == 4);
        assert (d[4] == 5);
    }

    #[test]
    fn test_init_fn() {
        // Test on-stack init_fn.
        let v = init_fn(3u, square);
        assert (len(v) == 3u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);

        // Test on-heap init_fn.
        v = init_fn(5u, square);
        assert (len(v) == 5u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);
        assert (v[3] == 9u);
        assert (v[4] == 16u);
    }

    #[test]
    fn test_init_elt() {
        // Test on-stack init_elt.
        let v = init_elt(2u, 10u);
        assert (len(v) == 2u);
        assert (v[0] == 10u);
        assert (v[1] == 10u);

        // Test on-heap init_elt.
        v = init_elt(6u, 20u);
        assert (v[0] == 20u);
        assert (v[1] == 20u);
        assert (v[2] == 20u);
        assert (v[3] == 20u);
        assert (v[4] == 20u);
        assert (v[5] == 20u);
    }

    #[test]
    fn test_is_empty() {
        assert (is_empty::<int>([]));
        assert (!is_empty([0]));
    }

    #[test]
    fn test_is_not_empty() {
        assert (is_not_empty([0]));
        assert (!is_not_empty::<int>([]));
    }

    #[test]
    fn test_head() {
        let a = [11, 12];
        assert (head(a) == 11);
    }

    #[test]
    fn test_tail() {
        let a = [11];
        assert (tail(a) == []);

        a = [11, 12];
        assert (tail(a) == [12]);
    }

    #[test]
    fn test_last() {
        let n = last_opt([]);
        assert (n == none);
        n = last_opt([1, 2, 3]);
        assert (n == some(3));
        n = last_opt([1, 2, 3, 4, 5]);
        assert (n == some(5));
    }

    #[test]
    fn test_slice() {
        // Test on-stack -> on-stack slice.
        let v = slice([1, 2, 3], 1u, 3u);
        assert (len(v) == 2u);
        assert (v[0] == 2);
        assert (v[1] == 3);

        // Test on-heap -> on-stack slice.
        v = slice([1, 2, 3, 4, 5], 0u, 3u);
        assert (len(v) == 3u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (v[2] == 3);

        // Test on-heap -> on-heap slice.
        v = slice([1, 2, 3, 4, 5, 6], 1u, 6u);
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
        let v = [1, 2, 3];
        let e = pop(v);
        assert (len(v) == 2u);
        assert (v[0] == 1);
        assert (v[1] == 2);
        assert (e == 3);

        // Test on-heap pop.
        v = [1, 2, 3, 4, 5];
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
        let v = [];
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
        let v = [];
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
        let v = [];
        grow_fn(v, 3u, square);
        assert (len(v) == 3u);
        assert (v[0] == 0u);
        assert (v[1] == 1u);
        assert (v[2] == 4u);
    }

    #[test]
    fn test_grow_set() {
        let v = [mutable 1, 2, 3];
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
        let v = [1u, 2u, 3u];
        let w = map(v, square_ref);
        assert (len(w) == 3u);
        assert (w[0] == 1u);
        assert (w[1] == 4u);
        assert (w[2] == 9u);

        // Test on-heap map.
        v = [1u, 2u, 3u, 4u, 5u];
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
        let v0 = [1, 2, 3, 4, 5];
        let v1 = [5, 4, 3, 2, 1];
        let u = map2::<int, int, int>(v0, v1, f);
        let i = 0;
        while i < 5 { assert (v0[i] * v1[i] == u[i]); i += 1; }
    }

    #[test]
    fn test_filter_map() {
        // Test on-stack filter-map.
        let v = [1u, 2u, 3u];
        let w = filter_map(v, square_if_odd);
        assert (len(w) == 2u);
        assert (w[0] == 1u);
        assert (w[1] == 9u);

        // Test on-heap filter-map.
        v = [1u, 2u, 3u, 4u, 5u];
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
        let all_even: [int] = [0, 2, 8, 6];
        let all_odd1: [int] = [1, 7, 3];
        let all_odd2: [int] = [];
        let mix: [int] = [9, 2, 6, 7, 1, 0, 0, 3];
        let mix_dest: [int] = [1, 3, 0, 0];
        assert (filter_map(all_even, halve) == map(all_even, halve_for_sure));
        assert (filter_map(all_odd1, halve) == []);
        assert (filter_map(all_odd2, halve) == []);
        assert (filter_map(mix, halve) == mix_dest);
    }

    #[test]
    fn test_filter() {
        assert filter([1u, 2u, 3u], is_odd) == [1u, 3u];
        assert filter([1u, 2u, 4u, 8u, 16u], is_three) == [];
    }

    #[test]
    fn test_foldl() {
        // Test on-stack fold.
        let v = [1u, 2u, 3u];
        let sum = foldl(0u, v, add);
        assert (sum == 6u);

        // Test on-heap fold.
        v = [1u, 2u, 3u, 4u, 5u];
        sum = foldl(0u, v, add);
        assert (sum == 15u);
    }

    #[test]
    fn test_foldl2() {
        fn sub(&&a: int, &&b: int) -> int {
            a - b
        }
        let v = [1, 2, 3, 4];
        let sum = foldl(0, v, sub);
        assert sum == -10;
    }

    #[test]
    fn test_foldr() {
        fn sub(&&a: int, &&b: int) -> int {
            a - b
        }
        let v = [1, 2, 3, 4];
        let sum = foldr(v, 0, sub);
        assert sum == -2;
    }

    #[test]
    fn test_iter_empty() {
        let i = 0;
        iter::<int>([], { |_v| i += 1 });
        assert i == 0;
    }

    #[test]
    fn test_iter_nonempty() {
        let i = 0;
        iter([1, 2, 3], { |v| i += v });
        assert i == 6;
    }

    #[test]
    fn test_iteri() {
        let i = 0;
        iteri([1, 2, 3], { |j, v|
            if i == 0 { assert v == 1; }
            assert j + 1u == v as uint;
            i += v;
        });
        assert i == 6;
    }

    #[test]
    fn test_riter_empty() {
        let i = 0;
        riter::<int>([], { |_v| i += 1 });
        assert i == 0;
    }

    #[test]
    fn test_riter_nonempty() {
        let i = 0;
        riter([1, 2, 3], { |v|
            if i == 0 { assert v == 3; }
            i += v
        });
        assert i == 6;
    }

    #[test]
    fn test_riteri() {
        let i = 0;
        riteri([0, 1, 2], { |j, v|
            if i == 0 { assert v == 2; }
            assert j == v as uint;
            i += v;
        });
        assert i == 3;
    }

    #[test]
    fn test_permute() {
        let results: [[int]];

        results = [];
        permute([]) {|v| results += [v]; }
        assert results == [[]];

        results = [];
        permute([7]) {|v| results += [v]; }
        assert results == [[7]];

        results = [];
        permute([1,1]) {|v| results += [v]; }
        assert results == [[1,1],[1,1]];

        results = [];
        permute([5,2,0]) {|v| results += [v]; }
        assert results == [[5,2,0],[5,0,2],[2,5,0],[2,0,5],[0,5,2],[0,2,5]];
    }

    #[test]
    fn test_any_and_all() {
        assert (any([1u, 2u, 3u], is_three));
        assert (!any([0u, 1u, 2u], is_three));
        assert (any([1u, 2u, 3u, 4u, 5u], is_three));
        assert (!any([1u, 2u, 4u, 5u, 6u], is_three));

        assert (all([3u, 3u, 3u], is_three));
        assert (!all([3u, 3u, 2u], is_three));
        assert (all([3u, 3u, 3u, 3u, 3u], is_three));
        assert (!all([3u, 3u, 0u, 1u, 2u], is_three));
    }

    #[test]
    fn test_any2_and_all2() {

        assert (any2([2u, 4u, 6u], [2u, 4u, 6u], is_equal));
        assert (any2([1u, 2u, 3u], [4u, 5u, 3u], is_equal));
        assert (!any2([1u, 2u, 3u], [4u, 5u, 6u], is_equal));
        assert (any2([2u, 4u, 6u], [2u, 4u], is_equal));

        assert (all2([2u, 4u, 6u], [2u, 4u, 6u], is_equal));
        assert (!all2([1u, 2u, 3u], [4u, 5u, 3u], is_equal));
        assert (!all2([1u, 2u, 3u], [4u, 5u, 6u], is_equal));
        assert (!all2([2u, 4u, 6u], [2u, 4u], is_equal));
    }

    #[test]
    fn test_zip_unzip() {
        let v1 = [1, 2, 3];
        let v2 = [4, 5, 6];

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
    fn test_position_elt() {
        assert position_elt([], 1) == none;

        let v1 = [1, 2, 3, 3, 2, 5];
        assert position_elt(v1, 1) == some(0u);
        assert position_elt(v1, 2) == some(1u);
        assert position_elt(v1, 5) == some(5u);
        assert position_elt(v1, 4) == none;
    }

    #[test]
    fn test_position() {
        fn less_than_three(&&i: int) -> bool { ret i < 3; }
        fn is_eighteen(&&i: int) -> bool { ret i == 18; }

        assert position([], less_than_three) == none;

        let v1 = [5, 4, 3, 2, 1];
        assert position(v1, less_than_three) == some(3u);
        assert position(v1, is_eighteen) == none;
    }

    #[test]
    fn test_position_from() {
        assert position_from([], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position_from(v, 0u, 0u, f) == none;
        assert position_from(v, 0u, 1u, f) == none;
        assert position_from(v, 0u, 2u, f) == some(1u);
        assert position_from(v, 0u, 3u, f) == some(1u);
        assert position_from(v, 0u, 4u, f) == some(1u);

        assert position_from(v, 1u, 1u, f) == none;
        assert position_from(v, 1u, 2u, f) == some(1u);
        assert position_from(v, 1u, 3u, f) == some(1u);
        assert position_from(v, 1u, 4u, f) == some(1u);

        assert position_from(v, 2u, 2u, f) == none;
        assert position_from(v, 2u, 3u, f) == none;
        assert position_from(v, 2u, 4u, f) == some(3u);

        assert position_from(v, 3u, 3u, f) == none;
        assert position_from(v, 3u, 4u, f) == some(3u);

        assert position_from(v, 4u, 4u, f) == none;
    }

    #[test]
    fn test_find() {
        assert find([], f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find(v, f) == some((1, 'b'));
        assert find(v, g) == none;
    }

    #[test]
    fn test_find_from() {
        assert find_from([], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert find_from(v, 0u, 0u, f) == none;
        assert find_from(v, 0u, 1u, f) == none;
        assert find_from(v, 0u, 2u, f) == some((1, 'b'));
        assert find_from(v, 0u, 3u, f) == some((1, 'b'));
        assert find_from(v, 0u, 4u, f) == some((1, 'b'));

        assert find_from(v, 1u, 1u, f) == none;
        assert find_from(v, 1u, 2u, f) == some((1, 'b'));
        assert find_from(v, 1u, 3u, f) == some((1, 'b'));
        assert find_from(v, 1u, 4u, f) == some((1, 'b'));

        assert find_from(v, 2u, 2u, f) == none;
        assert find_from(v, 2u, 3u, f) == none;
        assert find_from(v, 2u, 4u, f) == some((3, 'b'));

        assert find_from(v, 3u, 3u, f) == none;
        assert find_from(v, 3u, 4u, f) == some((3, 'b'));

        assert find_from(v, 4u, 4u, f) == none;
    }

    #[test]
    fn test_rposition() {
        assert find([], f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert position(v, f) == some(1u);
        assert position(v, g) == none;
    }

    #[test]
    fn test_rposition_from() {
        assert rposition_from([], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rposition_from(v, 0u, 0u, f) == none;
        assert rposition_from(v, 0u, 1u, f) == none;
        assert rposition_from(v, 0u, 2u, f) == some(1u);
        assert rposition_from(v, 0u, 3u, f) == some(1u);
        assert rposition_from(v, 0u, 4u, f) == some(3u);

        assert rposition_from(v, 1u, 1u, f) == none;
        assert rposition_from(v, 1u, 2u, f) == some(1u);
        assert rposition_from(v, 1u, 3u, f) == some(1u);
        assert rposition_from(v, 1u, 4u, f) == some(3u);

        assert rposition_from(v, 2u, 2u, f) == none;
        assert rposition_from(v, 2u, 3u, f) == none;
        assert rposition_from(v, 2u, 4u, f) == some(3u);

        assert rposition_from(v, 3u, 3u, f) == none;
        assert rposition_from(v, 3u, 4u, f) == some(3u);

        assert rposition_from(v, 4u, 4u, f) == none;
    }

    #[test]
    fn test_rfind() {
        assert rfind([], f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        fn g(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'd' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind(v, f) == some((3, 'b'));
        assert rfind(v, g) == none;
    }

    #[test]
    fn test_rfind_from() {
        assert rfind_from([], 0u, 0u, f) == none;

        fn f(xy: (int, char)) -> bool { let (_x, y) = xy; y == 'b' }
        let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

        assert rfind_from(v, 0u, 0u, f) == none;
        assert rfind_from(v, 0u, 1u, f) == none;
        assert rfind_from(v, 0u, 2u, f) == some((1, 'b'));
        assert rfind_from(v, 0u, 3u, f) == some((1, 'b'));
        assert rfind_from(v, 0u, 4u, f) == some((3, 'b'));

        assert rfind_from(v, 1u, 1u, f) == none;
        assert rfind_from(v, 1u, 2u, f) == some((1, 'b'));
        assert rfind_from(v, 1u, 3u, f) == some((1, 'b'));
        assert rfind_from(v, 1u, 4u, f) == some((3, 'b'));

        assert rfind_from(v, 2u, 2u, f) == none;
        assert rfind_from(v, 2u, 3u, f) == none;
        assert rfind_from(v, 2u, 4u, f) == some((3, 'b'));

        assert rfind_from(v, 3u, 3u, f) == none;
        assert rfind_from(v, 3u, 4u, f) == some((3, 'b'));

        assert rfind_from(v, 4u, 4u, f) == none;
    }

    #[test]
    fn reverse_and_reversed() {
        let v: [mutable int] = [mutable 10, 20];
        assert (v[0] == 10);
        assert (v[1] == 20);
        reverse(v);
        assert (v[0] == 20);
        assert (v[1] == 10);
        let v2 = reversed::<int>([10, 20]);
        assert (v2[0] == 20);
        assert (v2[1] == 10);
        v[0] = 30;
        assert (v2[0] == 20);
        // Make sure they work with 0-length vectors too.

        let v4 = reversed::<int>([]);
        assert (v4 == []);
        let v3: [mutable int] = [mutable];
        reverse::<int>(v3);
    }

    #[test]
    fn reversed_mut() {
        let v2 = reversed::<int>([mutable 10, 20]);
        assert (v2[0] == 20);
        assert (v2[1] == 10);
    }

    #[test]
    fn test_init() {
        let v = init([1, 2, 3]);
        assert v == [1, 2];
    }

    #[test]
    fn test_split() {
        fn f(&&x: int) -> bool { x == 3 }

        assert split([], f) == [];
        assert split([1, 2], f) == [[1, 2]];
        assert split([3, 1, 2], f) == [[], [1, 2]];
        assert split([1, 2, 3], f) == [[1, 2], []];
        assert split([1, 2, 3, 4, 3, 5], f) == [[1, 2], [4], [5]];
    }

    #[test]
    fn test_splitn() {
        fn f(&&x: int) -> bool { x == 3 }

        assert splitn([], 1u, f) == [];
        assert splitn([1, 2], 1u, f) == [[1, 2]];
        assert splitn([3, 1, 2], 1u, f) == [[], [1, 2]];
        assert splitn([1, 2, 3], 1u, f) == [[1, 2], []];
        assert splitn([1, 2, 3, 4, 3, 5], 1u, f) == [[1, 2], [4, 3, 5]];
    }

    #[test]
    fn test_rsplit() {
        fn f(&&x: int) -> bool { x == 3 }

        assert rsplit([], f) == [];
        assert rsplit([1, 2], f) == [[1, 2]];
        assert rsplit([1, 2, 3], f) == [[1, 2], []];
        assert rsplit([1, 2, 3, 4, 3, 5], f) == [[1, 2], [4], [5]];
    }

    #[test]
    fn test_rsplitn() {
        fn f(&&x: int) -> bool { x == 3 }

        assert rsplitn([], 1u, f) == [];
        assert rsplitn([1, 2], 1u, f) == [[1, 2]];
        assert rsplitn([1, 2, 3], 1u, f) == [[1, 2], []];
        assert rsplitn([1, 2, 3, 4, 3, 5], 1u, f) == [[1, 2, 3, 4], [5]];
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_init_empty() {
        init::<int>([]);
    }

    #[test]
    fn test_concat() {
        assert concat([[1], [2,3]]) == [1, 2, 3];
    }

    #[test]
    fn test_connect() {
        assert connect([], 0) == [];
        assert connect([[1], [2, 3]], 0) == [1, 0, 2, 3];
        assert connect([[1], [2], [3]], 0) == [1, 0, 2, 0, 3];
    }

    #[test]
    fn test_windowed () {
        assert [[1u,2u,3u],[2u,3u,4u],[3u,4u,5u],[4u,5u,6u]]
              == windowed (3u, [1u,2u,3u,4u,5u,6u]);

        assert [[1u,2u,3u,4u],[2u,3u,4u,5u],[3u,4u,5u,6u]]
              == windowed (4u, [1u,2u,3u,4u,5u,6u]);

        assert [] == windowed (7u, [1u,2u,3u,4u,5u,6u]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_windowed_() {
        let _x = windowed (0u, [1u,2u,3u,4u,5u,6u]);
    }

    #[test]
    fn to_mut_no_copy() unsafe {
        let x = [1, 2, 3];
        let addr = unsafe::to_ptr(x);
        let x_mut = to_mut(x);
        let addr_mut = unsafe::to_ptr(x_mut);
        assert addr == addr_mut;
    }

    #[test]
    fn from_mut_no_copy() unsafe {
        let x = [mut 1, 2, 3];
        let addr = unsafe::to_ptr(x);
        let x_imm = from_mut(x);
        let addr_imm = unsafe::to_ptr(x_imm);
        assert addr == addr_imm;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
