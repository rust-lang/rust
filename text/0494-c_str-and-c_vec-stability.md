- Start Date: 2015-01-02
- RFC PR: https://github.com/rust-lang/rfcs/pull/494
- Rust Issue: https://github.com/rust-lang/rust/issues/20444

# Summary

* Remove the `std::c_vec` module
* Move `std::c_str` under a new `std::ffi` module, not exporting the `c_str`
  module.
* Focus `CString` on *Rust-owned* bytes, providing a static assertion that a
  pile of bytes has no interior nuls but has a trailing nul.
* Provide convenience functions for translating *C-owned* types into slices in
  Rust.

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

In refactoring all usage could be categorized into one of three categories:

1. A Rust type wants to be passed into C.
2. A C type was handed to Rust, but Rust does not own it.
3. A C type was handed to Rust, and Rust owns it.

The current `CString` attempts to handle all three of these concerns all at
once, somewhat conflating desires. Additionally, `CVec` provides a fairly
different interface than `CString` while providing similar functionality.

## A new `std::ffi`

> **Note**: an old implementation of the design below can be found [in a branch
> of mine][c_str]

[c_str]: https://github.com/alexcrichton/rust/blob/cstr/src/librustrt/c_str.rs

The entire `c_str` module will be deleted as-is today and replaced with the
following interface at the new location `std::ffi`:

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

pub unsafe fn c_str_to_bytes<'a>(raw: &'a *const libc::c_char) -> &'a [u8] { /* ... */ }
pub unsafe fn c_str_to_bytes_with_nul<'a>(raw: &'a *const libc::c_char) -> &'a [u8] { /* ... */ }
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
  `ffi::c_str_to_bytes` plus one of `str::from_utf8` or
  `str::from_utf8_unchecked`.
* `String::from_raw_buf` - similarly to `from_c_str`, each step should be
  composed individually to perform the required checks. This would involve using
  `ffi::c_str_to_bytes`, `str::from_utf8`, and `.to_string()`.
* `String::from_raw_buf_len` - this should be replaced the same way as
  `String::from_raw_buf` except that `slice::from_raw_buf` is used instead of
  `ffi`.

## Removing `c_vec`

The new `ffi` module serves as a solution to desires (1) and (2) above, but
the third use case is left unsolved so far. This is what the current `c_vec`
module is attempting to solve, but it does so in a somewhat ad-hoc fashion. The
constructor for the type takes a `proc` destructor to invoke when the vector is
dropped to allow for custom destruction. To make matters a little more
interesting, the `CVec` type provides a default constructor which invokes
`libc::free` on the pointer.

Transferring ownership of pointers without a custom deallocation function is in
general quite a dangerous operation for libraries to perform. Not all platforms
support the ability to `malloc` in one library and `free` in the other, and this
is also generally considered an antipattern.

Creating a custom wrapper struct with a simple `Deref` and `Drop` implementation
as necessary is likely to be sufficient for this use case, so this RFC proposes
removing the entire `c_vec` module with no replacement. It is expected that a
utility crate for interoperating with raw pointers in this fashion may manifest
itself on crates.io, and inclusion into the standard library can be considered
at that time.

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

* The `from_vec_unchecked` function could do precisely 0 work instead of always
  pushing a 0 at the end.

# Unresolved questions

* On some platforms, `libc::c_char` is not necessarily just one byte, which
  these types rely on. It's unclear how much this should affect the design of
  this module as to how important these platforms are.

* Are the `*_with_nul` functions necessary on `CString`?
