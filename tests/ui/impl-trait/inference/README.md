# Inference tests on opaque types in defining items

When considering an opaque type within a body that is permitted to
define the hidden type for that opaque, we're expecting the new trait
solver to sometimes produce different results.

Sometimes this is due to the fact that the new trait solver considers
more cases to be defining, such as when defining the hidden type
allows an impl for some wrapper type to be matched.

In other cases, this is due to lazy normalization, which e.g. allows
the new trait solver to more consistently handle as a concrete type
the return value of...

```rust
id2<T>(_: T, x: T ) -> T { x }
```

...when it is called with a value of the opaque type and a value of
the corresponding hidden type.

However, the new trait solver is not yet done, and it does not
consistently produce the results we expect that it will once
complete.

As we work toward stabilizing type alias impl Trait (TAIT), we need to
plan for what the behavior of the new trait solver will be.
Similarly, since return position impl Trait (RPIT) is already stable
but will see inference differences with the new trait solver, we need
to begin to plan for that also.

To help enable this planning, this directory contains test cases that
define what the correct inference behavior should be when handling
opaue types.

Where the correct behavior does not match the behavior of either the
new or the old trait solver, we've chosen to mark that as a known
bug.  For the new solver, we've done this since it is still a work in
progress and is expected to eventually model the correct behavior.
For the old solver, we've done this since the behavior is inconsistent
and often surprising, and since we may need to add future-incompat
lints or take other steps to prepare the ecosystem for the transition
to the new solver.
