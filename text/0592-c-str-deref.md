- Start Date: 2015-01-17
- RFC PR: https://github.com/rust-lang/rfcs/pull/592
- Rust Issue: https://github.com/rust-lang/rust/issues/22469

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
DST, `[libc::c_char]`. The slice type as it is, however, is a poor choice
for representing borrowed C string data, since:

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

This proposal introduces `CStr`, a type to designate a null-terminated
string. This type does not implement `Sized`, `Copy`, or `Clone`.
References to `CStr` are only safely obtained by dereferencing `CString`
and a few other helper methods, described below. A `CStr` value should provide
no size information, as there is intent to turn `CStr` into an
[unsized type](https://github.com/rust-lang/rfcs/issues/813),
pending resolution on that proposal.

## Stage 1: CStr, a DST with a weight problem

As current Rust does not have unsized types that are not DSTs, at this stage
`CStr` is defined as a newtype over a character slice:

```rust
#[repr(C)]
pub struct CStr {
    chars: [libc::c_char]
}

impl CStr {
    pub fn as_ptr(&self) -> *const libc::c_char {
        self.chars.as_ptr()
    }
}
```

`CString` is changed to dereference to `CStr`:

```rust
impl Deref for CString {
    type Target = CStr;
    fn deref(&self) -> &CStr { ... }
}
```

In implementation, the `CStr` value needs a length for the internal slice.
This RFC provides no guarantees that the length will be equal to the length
of the string, or be any particular value suitable for safe use.

## Stage 2: unsized CStr

If unsized types are enabled later one way of another, the definition
of `CStr` would change to an unsized type with statically sized contents.
The authors of this RFC believe this would constitute no breakage to code
using `CStr` safely. With a view towards this future change, it's recommended
to avoid any unsafe code depending on the internal representation of `CStr`.

## Returning C strings

In cases when an FFI function returns a pointer to a non-owned C string,
it might be preferable to wrap the returned string safely as a 'thin'
`&CStr` rather than scan it into a slice up front. To facilitate this,
conversion from a raw pointer should be added (with an inferred lifetime
as per [the established convention](https://github.com/rust-lang/rfcs/pull/556)):
```rust
impl CStr {
    pub unsafe fn from_ptr<'a>(ptr: *const libc::c_char) -> &'a CStr {
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
a `CString` via auto-dereferencing.

## Remove c_str_to_bytes

The functions `c_str_to_bytes` and `c_str_to_bytes_with_nul`, with their
problematic lifetime semantics, are deprecated and eventually removed
in favor of composition of the functions described above:
`c_str_to_bytes(&ptr)` becomes `CStr::from_ptr(ptr).to_bytes()`.

## Proof of concept

The described interface changes are implemented in crate
[c_string](https://github.com/mzabaluev/rust-c-str).

# Drawbacks

The change of the deref target type is another breaking change to `CString`.
In practice the main purpose of borrowing from `CString` is to obtain a
raw pointer with `.as_ptr()`; for code which only does this and does not
expose the slice in type annotations, parameter signatures and so on,
the change should not be breaking since `CStr` also provides
this method.

Making the deref target unsized throws away the length information
intrinsic to `CString` and makes it less useful as a container for bytes.
This is countered by the fact that there are general purpose byte containers
in the core libraries, whereas `CString` addresses the specific need to
convey string data from Rust to C-style APIs.

# Alternatives

If the proposed enhancements or other equivalent facilities are not adopted,
users of Rust can turn to third-party libraries for better convenience
and safety when working with C strings. This may result in proliferation of
incompatible helper types in public APIs until a dominant de-facto solution
is established.

# Unresolved questions

Need a `Cow`?
