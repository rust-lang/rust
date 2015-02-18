% Data types

### Use custom types to imbue meaning; do not abuse `bool`, `Option` or other core types. **[FIXME: needs RFC]**

Prefer

```rust
let w = Widget::new(Small, Round)
```

over

```rust
let w = Widget::new(true, false)
```

Core types like `bool`, `u8` and `Option` have many possible interpretations.

Use custom types (whether `enum`s, `struct`, or tuples) to convey
interpretation and invariants. In the above example,
it is not immediately clear what `true` and `false` are conveying without
looking up the argument names, but `Small` and `Round` are more suggestive.

Using custom types makes it easier to expand the
options later on, for example by adding an `ExtraLarge` variant.

See [the newtype pattern](newtype.md) for a no-cost way to wrap
existing types with a distinguished name.

### Prefer private fields, except for passive data. **[FIXME: needs RFC]**

Making a field public is a strong commitment: it pins down a representation
choice, _and_ prevents the type from providing any validation or maintaining any
invariants on the contents of the field, since clients can mutate it arbitrarily.

Public fields are most appropriate for `struct` types in the C spirit: compound,
passive data structures. Otherwise, consider providing getter/setter methods
and hiding fields instead.

> **[FIXME]** Cross-reference validation for function arguments.

### Use custom `enum`s for alternatives, `bitflags` for C-style flags. **[FIXME: needs RFC]**

Rust supports `enum` types with "custom discriminants":

~~~~
enum Color {
  Red = 0xff0000,
  Green = 0x00ff00,
  Blue = 0x0000ff
}
~~~~

Custom discriminants are useful when an `enum` type needs to be serialized to an
integer value compatibly with some other system/language. They support
"typesafe" APIs: by taking a `Color`, rather than an integer, a function is
guaranteed to get well-formed inputs, even if it later views those inputs as
integers.

An `enum` allows an API to request exactly one choice from among many. Sometimes
an API's input is instead the presence or absence of a set of flags. In C code,
this is often done by having each flag correspond to a particular bit, allowing
a single integer to represent, say, 32 or 64 flags. Rust's `std::bitflags`
module provides a typesafe way for doing so.

### Phantom types. [FIXME]

> **[FIXME]** Add some material on phantom types (https://blog.mozilla.org/research/2014/06/23/static-checking-of-units-in-servo/)
