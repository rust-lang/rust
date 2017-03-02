- Start Date: 2015-02-19
- RFC PR: [rust-lang/rfcs#735](https://github.com/rust-lang/rfcs/pull/735)
- Rust Issue: [rust-lang/rust#22563](https://github.com/rust-lang/rust/issues/22563)

# Summary

Allow inherent implementations on types outside of the module they are defined in,
effectively reverting [RFC PR 155](https://github.com/rust-lang/rfcs/pull/155).

# Motivation

The main motivation for disallowing such `impl` bodies was the implementation
detail of fake modules being created to allow resolving `Type::method`, which
only worked correctly for `impl Type {...}` if a `struct Type` or `enum Type`
were defined in the same module. The old mechanism was obsoleted by UFCS,
which desugars `Type::method` to `<Type>::method` and perfoms a type-based
method lookup instead, with path resolution having no knowledge of inherent
`impl`s - and all of that was implemented by [rust-lang/rust#22172](https://github.com/rust-lang/rust/pull/22172).

Aside from invalidating the previous RFC's motivation, there is something to be
said about dealing with restricted inherent `impl`s: it leads to non-DRY single
use extension traits, the worst offender being `AstBuilder` in libsyntax, with
almost 300 lines of redundant method definitions.

# Detailed design

Remove the existing limitation, and only require that the `Self` type of the
`impl` is defined in the same crate. This allows moving methods to other modules:
```rust
struct Player;

mod achievements {
    struct Achievement;
    impl Player {
        fn achieve(&mut self, _: Achievement) {}
    }
}
```

# Drawbacks

Consistency and ease of finding method definitions by looking at the module the
type is defined in, has been mentioned as an advantage of this limitation.
However, trait `impl`s already have that problem and single use extension traits
could arguably be worse.

# Alternatives

- Leave it as it is. Seems unsatisfactory given that we're no longer limited
  by implementation details.

- We could go further and allow adding inherent methods to any type that could
  implement a trait outside the crate:
  ```rust
  struct Point<T> { x: T, y: T }
  impl<T: Float> (Vec<Point<T>>, T) {
      fn foo(&mut self) -> T { ... }
  }
  ```

  The implementation would reuse the same coherence rules as for trait `impl`s,
  and, for looking up methods, the "type definition to impl" map would be replaced
  with a map from method name to a set of `impl`s containing that method.

  *Technically*, I am not aware of any formulation that limits inherent methods
  to user-defined types in the same crate, and this extra support could turn out
  to have a straight-foward implementation with no complications, but I'm trying
  to present the whole situation to avoid issues in the future - even though I'm
  not aware of backwards compatibility ones or any related to compiler internals.

# Unresolved questions

None.
