% Traits

Traits are probably Rust's most complex feature, supporting a wide range of use
cases and design tradeoffs. Patterns of trait usage are still emerging.

### Know whether a trait will be used as an object. [FIXME: needs RFC]

Trait objects have some [significant limitations](objects.md): methods
invoked through a trait object cannot use generics, and cannot use
`Self` except in receiver position.

When designing a trait, decide early on whether the trait will be used
as an [object](objects.md) or as a [bound on generics](generics.md);
the tradeoffs are discussed in each of the linked sections.

If a trait is meant to be used as an object, its methods should take
and return trait objects rather than use generics.


### Default methods [FIXME]

> **[FIXME]** Guidelines for default methods.
