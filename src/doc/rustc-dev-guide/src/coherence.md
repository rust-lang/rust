# Coherence

> NOTE: this is based on [notes by @lcnr](https://github.com/rust-lang/rust/pull/121848)

Coherence checking is what detects both of trait impls and inherent impls overlapping with others.
(reminder: [inherent impls](https://doc.rust-lang.org/reference/items/implementations.html#inherent-implementations) are impls of concrete types like `impl MyStruct {}`)

Overlapping trait impls always produce an error,
while overlapping inherent impls result in an error only if they have methods with the same name.

Checking for overlaps is split in two parts. First there's the [overlap check(s)](#overlap-checks), 
which finds overlaps between traits and inherent implementations that the compiler currently knows about.

However, Coherence also results in an error if any other impls **could** exist,
even if they are currently unknown. 
This affects impls which may get added to upstream crates in a backwards compatible way,
and impls from downstream crates. 
This is called the Orphan check.

## Overlap checks

Overlap checks are performed for both inherent impls, and for trait impls.
This uses the same overlap checking code, really done as two separate analyses.
Overlap checks always consider pairs of implementations, comparing them to each other.

Overlap checking for inherent impl blocks is done through `fn check_item` (in coherence/inherent_impls_overlap.rs),
where you can very clearly see that (at least for small `n`), the check really performs `n^2`
comparisons between impls. 

In the case of traits, this check is currently done as part of building the [specialization graph](traits/specialization.md),
to handle specializing impls overlapping with their parent, but this may change in the future.

In both cases, all pairs of impls are checked for overlap.

Overlapping is sometimes partially allowed:

1. for marker traits
2. under [specialization](traits/specialization.md)

but normally isn't. 

The overlap check has various modes (see [`OverlapMode`]).
Importantly, there's the explicit negative impl check, and the implicit negative impl check.
Both try to prove that an overlap is definitely impossible.

[`OverlapMode`]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_middle/traits/specialization_graph/enum.OverlapMode.html

### The explicit negative impl check

This check is done in [`impl_intersection_has_negative_obligation`]. 

This check tries to find a negative trait implementation. 
For example:

```rust
struct MyCustomErrorType;

// both in your own crate
impl From<&str> for MyCustomErrorType {}
impl<E> From<E> for MyCustomErrorType where E: Error {}
```

In this example, we'd get:
`MyCustomErrorType: From<&str>` and `MyCustomErrorType: From<?E>`, giving `?E = &str`.

And thus, these two implementations would overlap.
However, libstd provides `&str: !Error`, and therefore guarantees that there 
will never be a positive implementation of `&str: Error`, and thus there is no overlap.

Note that for this kind of negative impl check, we must have explicit negative implementations provided.
This is not currently stable.

[`impl_intersection_has_negative_obligation`]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_trait_selection/traits/coherence/fn.impl_intersection_has_negative_obligation.html

### The implicit negative impl check

This check is done in [`impl_intersection_has_impossible_obligation`],
and does not rely on negative trait implementations and is stable.

Let's say there's a 
```rust
impl From<MyLocalType> for Box<dyn Error> {}  // in your own crate
impl<E> From<E> for Box<dyn Error> where E: Error {} // in std
```

This would give: `Box<dyn Error>: From<MyLocalType>`, and `Box<dyn Error>: From<?E>`,  
giving `?E = MyLocalType`.

In your crate there's no `MyLocalType: Error`, downstream crates cannot implement `Error` (a remote trait) for `MyLocalType` (a remote type).
Therefore, these two impls do not overlap.
Importantly, this works even if there isn't a `impl !Error for MyLocalType`.

[`impl_intersection_has_impossible_obligation`]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_trait_selection/traits/coherence/fn.impl_intersection_has_impossible_obligation.html

