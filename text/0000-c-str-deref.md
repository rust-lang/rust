- Start Date: 2015-01-17
- RFC PR: 
- Rust Issue: 

# Summary

Make `CString` dereference to a token type `CStr`, which designates
null-terminated string data.

```rust
// Type-checked to only accept C strings
fn safe_puts(s: &CStr) {
    unsafe { libc::puts(s.as_ptr()) };
}

fn main() {
    let s = CString::from_slice("A Rust string");
    safe_puts(s);
}
```

# Motivation

The type `std::ffi::CString` is used to prepare string data for passing
as null-terminated strings to FFI functions. This type dereferences to a
DST, `[libc::c_char]`. The slice type, however, is a poor choice for
representing borrowed C string data, since:

1. A slice does not express the C string invariant at compile time.
   Safe interfaces wrapping FFI functions cannot take slice references as is
   without dynamic checks (when null-terminated slices are expected) or
   building a temporary `CString` internally (in this case plain Rust slices
   must be passed with no interior NULs).
2. An allocated `CString` buffer is not the only desired source for
   borrowed C string data. Specifically, it should be possible to interpret
   a raw pointer, unsafely and at zero overhead, as a reference to a
   null-terminated string, so that the reference can then be used safely.
   However, in order to construct a slice (or a dynamically sized newtype
   wrapping a slice), its length has to be determined, which is unnecessary
   for the consuming FFI function that will only receive a thin pointer.
   Another likely data source are string and byte string literals: provided
   that a static string is null-terminated, there should be a way to pass it
   to FFI functions without an intermediate allocation in `CString`.

As a pattern of owned/borrowed type pairs has been established
thoughout other modules (see e.g.
[path reform](https://github.com/rust-lang/rfcs/pull/474)),
it makes sense that `CString` gets its own borrowed counterpart.

# Detailed design

## CStr, an Irrelevantly Sized Type

This proposal introduces `CStr`, a token type to designate a null-terminated
string. This type does not implement `Copy` or `Clone` and is only used in
borrowed references. `CStr` is sized, but its size and layout are of no
consequence to its users. It's only safely obtained by dereferencing
`CString` and a few other helper methods, described below.

```rust
#[repr(C)]
pub struct CStr {
    head: libc::c_char,
    marker: std::marker::NoCopy
}

impl CStr {
    pub fn as_ptr(&self) -> *const libc::c_char {
        &self.head as *const libc::c_char
    }
}

impl Deref for CString {
    type Target = CStr;
    fn deref(&self) -> &CStr { ... }
}
```

## Static C strings

A way to create `CStr` references from static Rust expressions asserted as
null-terminated string or byte slices is provided by a couple of functions:

```rust
impl CStr {
    pub fn from_static_bytes(bytes: &'static [u8]) -> &'static CStr { ... }
    pub fn from_static_str(s: &'static str) -> &'static CStr { ... }
}
```

As these functions mostly work with literals, they only assert that the
slice is terminated by a zero byte. It's the responsibility of the programmer
to ensure that the static data does not contain any unintended interior NULs
(the program will not crash, but the string will be interpreted up to the
first `'\0'` encountered). For non-literal data, `CStrBuf::from_bytes` or
`CStrBuf::from_vec` should be preferred.

## Returning C strings

In cases when an FFI function returns a pointer to a non-owned C string,
it might be preferable to wrap the returned string safely as a 'thin'
`&CStr` rather than scan it into a slice up front. To facilitate this,
conversion from a raw pointer should be added (with an inferred lifetime
as per another proposed [RFC](https://github.com/rust-lang/rfcs/pull/556)):
```rust
impl CStr {
    pub unsafe fn from_raw<'a>(ptr: *const libc::c_char) -> &'a CStr {
        ...
    }
}
```

For getting a slice out of a `CStr` reference, method `to_bytes` is
provided. The name is preferred over `as_bytes` to reflect the linear cost
of calculating the length.
```rust
impl CStr {
    pub fn to_bytes(&self) -> &[u8] { ... }
    pub fn to_bytes_with_nul(&self) -> &[u8] { ... }
}
```

An odd consequence is that it is valid, if wasteful, to call `to_bytes` on
`CString` via auto-dereferencing.

## Proof of concept

The described changes are implemented in crate
[c_string](https://github.com/mzabaluev/rust-c-str/tree/v0.3.0).

# Drawbacks

The change of the deref type is another breaking change to `CString`.
In practice the main purpose of borrowing from `CString` is to obtain a
raw pointer with `.as_ptr()`; for code which only does this and does not
expose the slice in type annotations, parameter signatures and so on,
the change should not be breaking since `CStr` also provides
this method.

While it's not possible outside of unsafe code to unintentionally copy out
or modify the nominal value of `CStr` under an immutable reference, some
unforeseen trouble or confusion can arise due to the structure having a
bogus size. A separate [RFC PR](https://github.com/rust-lang/rfcs/issues/709),
if accepted, will solve this by opting out of `Sized`.

# Alternatives

`CStr` could be made a newtype on DST `[libc::c_char]`, allowing no-cost
slices. It's not clear if this is useful, and the need to calculate length
up front might prevent some optimized uses possible with the 'thin'
reference.

If the proposed enhancements or other equivalent facilities are not adopted,
users of Rust can turn to third-party libraries for better convenience
and safety when working with C strings. This may result in proliferation of
incompatible helper types in public APIs until a dominant de-facto solution
is established.

# Unresolved questions

The present function `c_str_to_bytes(&ptr)` may be deprecated in favor of
the more composable `CStr::from_raw(ptr).to_bytes()`.

`CStr` can be made a
[truly unsized type](https://github.com/rust-lang/rfcs/issues/709),
pending on that proposal's approval.

Need a `Cow`?
