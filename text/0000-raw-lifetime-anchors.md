- Start Date: 2015-01-06
- RFC PR:
- Rust Issue:

# Summary

Provide more flexible, less error-prone lifetime support
for unsafe pointer conversions throughout the libraries.

The currently tedious:
```rust
let s = std::slice::from_raw_buf(&ptr, len)
std::mem::copy_lifetime(self, s)
```
Becomes:
```rust
std::slice::from_raw_buf_with_lifetime(ptr, len, self)
```
Or, after a breaking change of the current API:
```rust
std::slice::from_raw_buf(ptr, len, self)
```

Static lifetime can be easy too:
```rust
std::slice::from_raw_buf(ptr, len, std::mem::STATIC)
```

# Motivation

The current library convention on functions constructing borrowed
values from raw pointers has the pointer passed by reference, which
reference's lifetime is carried over to the return value.
Unfortunately, the lifetime of a raw pointer is often not indicative
of the lifetime of the pointed-to data. This may lead to mistakes
since the acquired reference crosses into the "safe" domain without
much indication in the code, while the pointed-to value may become
invalid at any time.

A typical use case where the lifetime needs to be adjusted is
in bindings to a foregn library, when returning a reference to an object's
inner value (we know from the library's API contract that
the inner data's lifetime is bound to the containing object):
```rust
impl Outer {
    fn inner_str(&self) -> &[u8] {
        unsafe {
            let p = ffi::outer_get_inner_str(&self.raw);
            let s = std::slice::from_raw_buf(p, libc::strlen(p));
            std::mem::copy_lifetime(self, s)
        }
    }
}
```

And here's a plausible gotcha:
```rust
let foo = unsafe { ffi::new_foo() };
let s = unsafe { std::slice::from_raw_buf(&foo.data, foo.len) };
// s lives as long as foo

// some lines later

unsafe { ffi::free_foo(foo) };

// more lines later, in perfectly safe-looking code:

let guess_what = s[0];
```

# Detailed design

The signature of `from_raw*` constructors is changed: the raw pointer is
passed by value, and a generic reference argument is appended to work as a
lifetime anchor (the value can be anything and is ignored):

```rust
fn from_raw_buf_with_lifetime<'a, T, Sized? U>(ptr: *const T, len: uint,
                                               life_anchor: &'a U)
                                              -> &'a [T]
```
```rust
fn from_raw_mut_buf_with_lifetime<'a, T, Sized? U>(ptr: *mut T, len: uint,
                                                   life_anchor: &'a U)
                                                  -> &'a mut [T]
```

The existing constructors can be deprecated, to open a migration
path towards reusing their shorter names with the new signatures
when the time to break the API is right.

The current usage can be mechanically changed.

```rust
let s = std::slice::from_raw_buf(&ptr, len)
```
becomes:
```rust
std::slice::from_raw_buf_with_lifetime(ptr, len, &ptr)
```
However, it's better to try to find a more appropriate lifetime anchor
for each use.

## Fix copy_mut_lifetime

While we are at it, the first parameter of `std::mem::copy_mut_lifetime`
could be made a non-mutable reference. There is no reason for the lifetime
anchor to be mutable: the pointer's mutability is usually the relevant
question, and it's an unsafe function to begin with. This wart may
breed tedious, mut-happy, or transmute-happy code, when e.g. a container
providing the lifetime for a mutable view into its contents is not itself
necessarily mutable.

## Anchor for static references

To facilitate conversion of pointers to static references, an anchor constant
is provided in `std::mem`:

```rust
pub const STATIC: &'static () = &();
```

A simple `""` works just as well, while not being as descriptive
regarding its purpose.

# Drawbacks

The proposal adds new functions to the library for something that is
already doable, albeit with some inconvenience. Replacement of the existing
constructors, if approved, is a breaking change.

# Alternatives

The `from_raw*` functions can lose input-derived lifetimes altogether,
reverting to an earlier design:
```rust
pub unsafe fn from_raw_buf<'a, T>(ptr: *const T, len: usize) -> &'a T
```
Such functions would be usable without explicit type annotation and unelided
lifetime parameters only in another function's return value context. For other
uses, wrapper functions would often be created as workarounds.

The status quo convention can be used despite its problems and inconvenience.

# Unresolved questions

Should the existing by-reference constructors be deprecated and eventually
removed? If yes, should this change be done before 1.0?
