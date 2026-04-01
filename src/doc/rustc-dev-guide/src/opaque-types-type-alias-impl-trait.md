# Opaque types (type alias `impl Trait`)

Opaque types are syntax to declare an opaque type alias that only
exposes a specific set of traits as their interface; the concrete type in the
background is inferred from a certain set of use sites of the opaque type.

This is expressed by using `impl Trait` within type aliases, for example:

```rust,ignore
type Foo = impl Bar;
```

This declares an opaque type named `Foo`, of which the only information is that
it implements `Bar`. Therefore, any of `Bar`'s interface can be used on a `Foo`,
but nothing else (regardless of whether the concrete type implements any other traits).

Since there needs to be a concrete background type,
you can (as of <!-- date-check --> May 2025) express that type
by using the opaque type in a "defining use site".

```rust,ignore
struct Struct;
impl Bar for Struct { /* stuff */ }
#[define_opaque(Foo)]
fn foo() -> Foo {
    Struct
}
```

Any other "defining use site" needs to produce the exact same type.

Note that defining a type alias to an opaque type is an unstable feature.
To use it, you need `nightly` and the annotations `#![feature(type_alias_impl_trait)]` on the file and `#[define_opaque(Foo)]` on the method that links the opaque type to the concrete type.
Complete example:

```rust
#![feature(type_alias_impl_trait)]

trait Bar { /* stuff */ }

type Foo = impl Bar;

struct Struct;

impl Bar for Struct { /* stuff */ }

#[define_opaque(Foo)]
fn foo() -> Foo {
    Struct
}
```

## Defining use site(s)

Currently only the return value of a function can be a defining use site
of an opaque type (and only if the return type of that function contains
the opaque type).

The defining use of an opaque type can be any code *within* the parent
of the opaque type definition. This includes any siblings of the
opaque type and all children of the siblings.

The initiative for *"not causing fatal brain damage to developers due to
accidentally running infinite loops in their brain while trying to
comprehend what the type system is doing"* has decided to disallow children
of opaque types to be defining use sites.

### Associated opaque types

Associated opaque types can be defined by any other associated item
on the same trait `impl` or a child of these associated items. For instance:

```rust,ignore
trait Baz {
    type Foo;
    fn foo() -> Self::Foo;
}

struct Quux;

impl Baz for Quux {
    type Foo = impl Bar;
    fn foo() -> Self::Foo { ... }
}
```

For this you would also need to use `nightly` and the (different) `#![feature(impl_trait_in_assoc_type)]` annotation.
Note that you don't need a `#[define_opaque(Foo)]` on the method anymore as the opaque type is mentioned in the function signature (behind the associated type).
Complete example:

```
#![feature(impl_trait_in_assoc_type)]

trait Bar {}
struct Zap;

impl Bar for Zap {}

trait Baz {
    type Foo;
    fn foo() -> Self::Foo;
}

struct Quux;

impl Baz for Quux {
    type Foo = impl Bar;
    fn foo() -> Self::Foo { Zap }
}
```
