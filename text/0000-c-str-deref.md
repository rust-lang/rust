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
    safe_puts(c_str!("Look ma, a `&'static CStr` from a literal!"));
}
```

# Motivation

The type `std::ffi::CString` is used to prepare string data for passing
as null-terminated strings to FFI functions. This type dereferences to a
DST, `[libc::c_char]`. The DST, however, is a poor choice for representing
borrowed C string data, since:

1. The slice does not enforce the C string invariant at compile time.
   Safe interfaces wrapping FFI functions cannot take slice references as is
   without dynamic checks (when null-terminated slices are expected) or
   building a temporary `CString` internally (in this case plain Rust slices
   must be passed with no interior NULs). `CString`, for its part, is an
   owning container and is not convenient for passing by reference. A string
   literal, for example, would require a `CString` constructed from it at
   runtime to pass into a function expecting `&CString`.
2. The primary consumers of the borrowed pointers, FFI functions, do not care
   about the 'sized' aspect of the DST. The borrowed reference is
   therefore needlessly 'fat' for its primary purpose.

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

## c_str!

For added convenience in passing literal string data to FFI functions,
a macro is provided that appends a literal with `"\0"` and returns it
as `&'static CStr`:
```rust
#[macro_export]
macro_rules! c_str {
    ($lit:expr) => {
        $crate::ffi::CStr::from_static_str(concat!($lit, "\0"))
    }
}
```
Going forward, it would be good to make `c_str!` also accept byte strings
on input, through a [byte string concatenation
macro](https://github.com/rust-lang/rfcs/pull/566). Ultimately, it could be
made workable in static expressions through a compiler plugin.

## Returning C strings

In cases when an FFI function returns a pointer to a non-owned C string,
it might be preferable to wrap the returned string safely as a 'thin'
`&CStr` rather than scan it into a slice up front. To facilitate this,
conversion from a raw pointer should be added:
```rust
impl CStr {
    pub unsafe fn from_raw_ptr<'a>(ptr: *const libc::c_char) -> &'a CStr
    { ... }
}
```

For getting a slice out of a `CStr` reference, method `parse_as_bytes` is
provided. The name is chosen to reflect the linear cost of calculating the
length.
```rust
impl CStr {
    pub fn parse_as_bytes(&self) -> &[u8] { ... }
}
```

An odd consequence is that it is valid, if wasteful, to call
`parse_as_bytes` on `CString` via auto-dereferencing.

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

`CStr` can be made a
[truly unsized type](https://github.com/rust-lang/rfcs/issues/709),
pending on that proposal's approval.

Need a `Cow`?
