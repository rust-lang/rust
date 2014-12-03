- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Stabilize the `std::{c_str, c_vec}` modules by re-working their interfaces and
refocusing each primitive for one particular task. The three broad categories of
interoperating with C will work via:

1. If you have a Rust string/byte slice which needs to be given to C, then the
   `CString` type will be used to statically guarantee that a terminating nul
   character and no interior nuls exist.

2. If C hands you a string which you want to inspect, but not own, then a helper
   function will assist in converting the C string to a byte slice.

3. If C hands you a string which you want to inspect and own, then a helper type
   will consume ownership and will act as a `Box<[u8]>` in essence.

# Motivation

The primary motivation for this RFC is to work out the stabilization of the
`c_str` and `c_vec` modules. Both of these modules exist for interoperating with
C types to ensure that values can cross the boundary of Rust and C relatively
safely. These types also need to be designed with ergonomics in mind to ensure
that it's tough to get them wrong and easy to get them right.

The current `CString` and `CVec` types are quite old and are long due for a
scrutinization, and these types are currently serving a number of competing
concerns:

1. A `CString` can both take ownership of a pointer as well as inspect a
   pointer.
2. A `CString` is always allocated/deallocated on the libc heap.
3. A `CVec` looks like a slice but does not quite act like one.
4. A `CString` looks like a byte slice but does not quite act like one.
5. There are a number of pieces of duplicated functionality throughout the
   standard library when dealing with raw C types. There are a number of
   conversion functions on the `Vec` and `String` types as well as the `str` and
   `slice` modules.

In general all of this functionality needs to be reconciled with one another to
provide a consistent and coherence interface when operating with types
originating from C.

# Detailed design

In refactoring `c_str` and `c_vec`, all usage could be categorized into one of
three categories:

1. A Rust type wants to be passed into C.
2. A C type was handed to Rust, but Rust does not own it.
3. A C type was handed to Rust, and Rust owns it.

The current `CString` attempts to handle all three of these concerns all at
once, somewhat conflating desires. Additionally, `CVec` provides a fairly
different interface than `CString` while providing similar functionality.

## A new `std::c_string`

> **Note**: an implementation of the design below can be found [in a branch of
> mine][c_str]

[c_str]: https://github.com/alexcrichton/rust/blob/cstr/src/librustrt/c_str.rs

The entire `c_str` module will be deleted as-is today and replaced with the
following interface at the new location `std::c_string`:

```rust
#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct CString { /* ... */ }

impl CString {
    pub fn from_slice(s: &[u8]) -> CString { /* ... */ }
    pub fn from_vec(s: Vec<u8>) -> CString { /* ... */ }
    pub unsafe fn from_vec_unchecked(s: Vec<u8>) -> CString { /* ... */ }

    pub fn as_slice(&self) -> &[libc::c_char] { /* ... */ }
    pub fn as_slice_with_nul(&self) -> &[libc::c_char] { /* ... */ }
    pub fn as_bytes(&self) -> &[u8] { /* ... */ }
    pub fn as_bytes_with_nul(&self) -> &[u8] { /* ... */ }
}

impl Deref<[libc::c_char]> for CString { /* ... */ }
impl Show for CString { /* ... */ }

pub unsafe fn from_raw_buf<'a>(raw: &'a *const libc::c_char) -> &'a [u8] { /* ... */ }
pub unsafe fn from_raw_buf_with_nul<'a>(raw: &'a *const libc::c_char) -> &'a [u8] { /* ... */ }
```

The new `CString` API is focused solely on providing a static assertion that a
byte slice contains no interior nul bytes and there is a terminating nul byte.
A `CString` is usable as a slice of `libc::c_char` similar to how a `Vec` is
usable as a slice, but a `CString` can also be viewed as a byte slice with a
concrete `u8` type. The default of `libc::c_char` was chosen to ensure that
`.as_ptr()` returns a pointer of the right value. Note that `CString` does not
provide a `DerefMut` implementation to maintain the static guarantee that there
are no interior nul bytes.

### Constructing a `CString`

One of the major departures from today's API is how a `CString` is constructed.
Today this can be done through the `CString::new` function or the `ToCStr`
trait. These two construction vectors serve two very different purposes, one for
C-originating data and one for Rust-originating data. This redesign of `CString`
is solely focused on going from Rust to C (case 1 above) and only supports
constructors in this flavor.

The first constructor, `from_slice`, is intended to allow `CString` to implement
an on-the-stack buffer optimization in the future without having to resort to a
`Vec` with its allocation. This is similar to the optimization performed by
`with_c_str` today. Of the other two constructors, `from_vec` will consume a
vector, assert there are no 0 bytes, an then push a 0 byte on the end. The
`from_vec_unchecked` constructor will not perform the verification, but will
still push a zero. Note that both of these constructors expose the fact that a
`CString` is not necessarily valid UTF-8.

The `ToCStr` trait is removed entirely (including from the prelude) in favor of
these construction functions. This could possibly be re-added in the future, but
for now it will be removed from the module.

### Working with `*const libc::c_char`

Instead of using `CString` to look at a `*const libc::c_char`, the module now
provides two conversion functions to go from a C string to a byte slice. The
signature of this function is similar to the new `std::slice::from_raw_buf`
function and will use the lifetime of the pointer itself as an anchor for the
lifetime of the returned slice.

These two functions solve the use case (2) above where a C string just needs to
be inspected. Because a C string is fundamentally just a pile of bytes, it's
interpreted in Rust as a `u8` slice. With these two functions, all of the
following functions will also be deprecated:

* `std::str::from_c_str` - this function should be replaced with
  `c_str::from_raw_buf` plus one of `str::from_utf8` or
  `str::from_utf8_unchecked`.
* `String::from_raw_buf` - similarly to `from_c_str`, each step should be
  composed individually to perform the required checks. This would involve using
  `c_str::from_raw_buf`, `str::from_utf8`, and `.to_string()`.
* `String::from_raw_buf_len` - this should be replaced the same way as
  `String::from_raw_buf` except that `slice::from_raw_buf` is used instead of
  `c_str`.

## A new `c_vec`

> **Note**: an implementation of the design below can be found [in a branch of
> mine][c_vec]

[c_vec]: https://github.com/alexcrichton/rust/blob/cstr/src/libstd/c_vec.rs

The new `c_str` module serves as a solution to desires (1) and (2) above, but
the third use case is left unsolved so far. This is what the new `c_vec` module
will be realized to do. The new module will look like:

```rust
pub struct CVec<T, D = LibcDtor> { /* ... */ }

impl<T> CVec<T> {
    pub unsafe fn new(base: *mut T, len: uint) -> CVec<T> { /* ... */ }
}
impl<u8> CVec<u8> {
    pub unsafe fn from_c_str(base: *mut libc::c_char) -> CVec<u8> { /* ... */ }
}
impl<T, D: Dtor<T>> CVec<T, D> {
    pub unsafe fn new_with_dtor(base: *mut T, len: uint, dtor: D) -> CVec<T, D> { /* ... */ }
}

impl<T, D> Deref<[T]> for CVec<T, D> { /* ... */ }
impl<T, D> DerefMut<[T]> for CVec<T, D> { /* ... */ }

pub trait Dtor<T> {
    fn destroy(&mut self, ptr: *mut T, len: uint);
}

pub struct LibcDtor;

impl<T> Dtor<T> for LibcDtor { /* ... */ }
```

The new `CVec` type is similar to the `CString` type in that it will provide a
few constructor functions, but largely rely on its `Deref` implementations to
inherit most of its methods. Fundamentally a `CVec` is quite similar to a
`Box<[T]>` with a different deallocation strategy. This is realized with the few
constructor functions, the `Deref`/`DerefMut` implementation, and the destructor
type parameter.

Each `CVec` will by default be deallocated with `libc::free` (what `LibcDtor`
does), but custom deallocation strategies can be implemented via the `Dtor`
trait.

### Construction

A `CVec` is primarily constructed through the `new` function where a pointer/len
pair is specified. The constructor consumes ownership of the memory and will
treat it as mutable (hence `*mut T`). The default constructor is also hardwired
to `LibcDtor` in a similar fashion to `HashMap::new` being hardwired to sip
hashing.

A convenience constructor, `from_c_str`, is provided for taking ownership of a
foreign-allocated C string. The returned vector is a `u8` vector to indicate
that it is intended for use in Rust itself (with a Rust type, not
`libc::c_char`).

## Working with C Strings

The design above has been implemented in [a branch][branch] of mine where the
fallout can be seen. The primary impact of this change is that the `to_c_str`
and `with_c_str` methods are no longer in the prelude by default, and
`CString::from_*` must be called in order to create a C string.

[branch]: https://github.com/alexcrichton/rust/tree/cstr

# Drawbacks

* Whenever Rust works with a C string, it's tough to avoid the cost associated
  with the initial length calculation. All types provided here involve
  calculating the length of a C string up front, and no type is provided to
  operate on a C string without calculating its length.

* With the removal of the `ToCStr` trait, unnecessary allocations may be made
  when converting to a `CString`. For example, a `Vec<u8>` can be called by
  directly calling `CString::from_vec`, but it may be more frequently called via
  `CString::from_slice`, resulting in an unnecessary allocation. Note, however,
  that one would have to remember to call `into_c_str` on the `ToCStr` trait, so
  it doesn't necessarily help too too much.

* The ergonomics of operating C strings have been somewhat reduced as part of
  this design. The `CString::from_slice` method is somewhat long to call
  (compared to `to_c_string`), and convenience methods of going straight from a
  `*const libc::c_char` were deprecated in favor of only supporting a conversion
  to a slice.

# Alternatives

* There is an [alternative RFC](https://github.com/rust-lang/rfcs/pull/435)
  which discusses pursuit of today's general design of the `c_str` module  as
  well as a refinement of its current types.

* The `Dtor` trait in the `c_vec` module could possibly be replaced with what it
  is today, a `proc`. This imposes a requirement of `Send`, however, and cannot
  be 0-size. It is not clear, however, whether using a trait for these two
  reasons is worth it.

* The `from_vec_unchecked` function could do precisely 0 work instead of always
  pushing a 0 at the end.

# Unresolved questions

* On some platforms, `libc::c_char` is not necessarily just one byte, which
  these types rely on. It's unclear how much this should affect the design of
  this module as to how important these platforms are.

* Are the `*_with_nul` functions necessary on `CString`?
