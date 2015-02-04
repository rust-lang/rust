- Start Date: 2015-01-06
- RFC PR: [rust-lang/rfcs#556](https://github.com/rust-lang/rfcs/pull/556)
- Rust Issue: [rust-lang/rust#21923](https://github.com/rust-lang/rust/issues/21923)

# Summary

Establish a convention throughout the core libraries for unsafe functions
constructing references out of raw pointers. The goal is to improve usability
while promoting awareness of possible pitfalls with inferred lifetimes.

# Motivation

The current library convention on functions constructing borrowed
values from raw pointers has the pointer passed by reference, which
reference's lifetime is carried over to the return value.
Unfortunately, the lifetime of a raw pointer is often not indicative
of the lifetime of the pointed-to data. So the status quo eschews the
flexibility of inferring the lifetime from the usage, while falling short
of providing useful safety semantics in exchange.

A typical case where the lifetime needs to be adjusted is in bindings
to a foregn library, when returning a reference to an object's
inner value (we know from the library's API contract that
the inner data's lifetime is bound to the containing object):
```rust
impl Outer {
    fn inner_str(&self) -> &[u8] {
        unsafe {
            let p = ffi::outer_get_inner_str(&self.raw);
            let s = std::slice::from_raw_buf(&p, libc::strlen(p));
            std::mem::copy_lifetime(self, s)
        }
    }
}
```
Raw pointer casts also discard the lifetime of the original pointed-to value.

# Detailed design

The signature of `from_raw*` constructors will be changed back to what it
once was, passing a pointer by value:
```rust
unsafe fn from_raw_buf<'a, T>(ptr: *const T, len: uint) -> &'a [T]
```
The lifetime on the return value is inferred from the call context.

The current usage can be mechanically changed, while keeping an eye on
possible lifetime leaks which need to be worked around by e.g. providing
safe helper functions establishing lifetime guarantees, as described below.

## Document the unsafety

In many cases, the lifetime parameter will come annotated or elided from the
call context. The example above, adapted to the new convention, is safe
despite lack of any explicit annotation:
```rust
impl Outer {
    fn inner_str(&self) -> &[u8] {
        unsafe {
            let p = ffi::outer_get_inner_str(&self.raw);
            std::slice::from_raw_buf(p, libc::strlen(p))
        }
    }
}
```

In other cases, the inferred lifetime will not be correct:
```rust
    let foo = unsafe { ffi::new_foo() };
    let s = unsafe { std::slice::from_raw_buf(foo.data, foo.len) };

    // Some lines later
    unsafe { ffi::free_foo(foo) };

    // More lines later
    let guess_what = s[0];
    // The lifetime of s is inferred to extend to the line above.
    // That code told you it's unsafe, didn't it?
```

Given that the function is unsafe, the code author should exercise due care
when using it. However, the pitfall here is not readily apparent at the
place where the invalid usage happens, so it can be easily committed by an
inexperienced user, or inadvertently slipped in with a later edit.

To mitigate this, the documentation on the reference-from-raw functions
should include caveats warning about possible misuse and suggesting ways to
avoid it. When an 'anchor' object providing the lifetime is available, the
best practice is to create a safe helper function or method, taking a
reference to the anchor object as input for the lifetime parameter, like in
the example above. The lifetime can also be explicitly assigned with
`std::mem::copy_lifetime` or `std::mem::copy_lifetime_mut`, or annotated when
possible.

## Fix copy_mut_lifetime

To improve composability in cases when the lifetime does need to be assigned
explicitly, the first parameter of `std::mem::copy_mut_lifetime`
should be made an immutable reference. There is no reason for the lifetime
anchor to be mutable: the pointer's mutability is usually the relevant
question, and it's an unsafe function to begin with. This wart may
breed tedious, mut-happy, or transmute-happy code, when e.g. a container
providing the lifetime for a mutable view into its contents is not itself
necessarily mutable.

# Drawbacks

The implicitly inferred lifetimes are unsafe in sneaky ways, so care is
required when using the borrowed values.

Changing the existing functions is an API break.

# Alternatives

An earlier revision of this RFC proposed adding a generic input parameter to
determine the lifetime of the returned reference:
```rust
unsafe fn from_raw_buf<'a, T, U: Sized?>(ptr: *const T, len: uint,
                                         life_anchor: &'a U)
                                        -> &'a [T]
```
However, an object with a suitable lifetime is not always available
in the context of the call. In line with the general trend in Rust libraries
to favor composability, `std::mem::copy_lifetime` and
`std::mem::copy_lifetime_mut` should be the principal methods to explicitly
adjust a lifetime.

# Unresolved questions

Should the change in function parameter signatures be done before 1.0?

# Acknowledgements

Thanks to Alex Crichton for shepherding this proposal in a constructive and
timely manner. He has in fact rationalized the convention in its present form.
