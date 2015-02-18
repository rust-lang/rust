% Common traits

### Eagerly implement common traits. [FIXME: needs RFC]

Rust's trait system does not allow _orphans_: roughly, every `impl` must live
either in the crate that defines the trait or the implementing
type. Consequently, crates that define new types should eagerly implement all
applicable, common traits.

To see why, consider the following situation:

* Crate `std` defines trait `Show`.
* Crate `url` defines type `Url`, without implementing `Show`.
* Crate `webapp` imports from both `std` and `url`,

There is no way for `webapp` to add `Show` to `url`, since it defines neither.
(Note: the newtype pattern can provide an efficient, but inconvenient
workaround; see [newtype for views](../types/newtype.md))

The most important common traits to implement from `std` are:

```rust
Clone, Show, Hash, Eq
```

#### When safe, derive or otherwise implement `Send` and `Share`. [FIXME]

> **[FIXME]**. This guideline is in flux while the "opt-in" nature of
> built-in traits is being decided. See https://github.com/rust-lang/rfcs/pull/127

### Prefer to derive, rather than implement. [FIXME: needs RFC]

Deriving saves implementation effort, makes correctness trivial, and
automatically adapts to upstream changes.

### Do not overload operators in surprising ways. [FIXME: needs RFC]

Operators with built in syntax (`*`, `|`, and so on) can be provided for a type
by implementing the traits in `core::ops`. These operators come with strong
expectations: implement `Mul` only for an operation that bears some resemblance
to multiplication (and shares the expected properties, e.g. associativity), and
so on for the other traits.

### The `Drop` trait

The `Drop` trait is treated specially by the compiler as a way of
associating destructors with types. See
[the section on destructors](../../ownership/destructors.md) for
guidance.

### The `Deref`/`DerefMut` traits

#### Use `Deref`/`DerefMut` only for smart pointers. [FIXME: needs RFC]

The `Deref` traits are used implicitly by the compiler in many circumstances,
and interact with method resolution. The relevant rules are designed
specifically to accommodate smart pointers, and so the traits should be used
only for that purpose.

#### Do not fail within a `Deref`/`DerefMut` implementation. [FIXME: needs RFC]

Because the `Deref` traits are invoked implicitly by the compiler in sometimes
subtle ways, failure during dereferencing can be extremely confusing. If a
dereference might not succeed, target the `Deref` trait as a `Result` or
`Option` type instead.

#### Avoid inherent methods when implementing `Deref`/`DerefMut` [FIXME: needs RFC]

The rules around method resolution and `Deref` are in flux, but inherent methods
on a type implementing `Deref` are likely to shadow any methods of the referent
with the same name.
