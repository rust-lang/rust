% Changes with unclear timing

### Associated items

* Many traits that currently take type parameters should instead use associated
  types; this will _drastically_ simplify signatures in some cases.

* Associated constants would be useful in a few places, e.g. traits for
  numerics, traits for paths.

### Anonymous, unboxed return types (aka `impl Trait` types)

* See https://github.com/rust-lang/rfcs/pull/105

* Could affect API design in several places, e.g. the `Iterator` trait.

### Default type parameters

We are already using this in a few places (e.g. `HashMap`), but it's
feature-gated.

### Compile-time function evaluation (CTFE)

https://github.com/mozilla/rust/issues/11621

### Improved constant folding

https://github.com/rust-lang/rust/issues/7834
