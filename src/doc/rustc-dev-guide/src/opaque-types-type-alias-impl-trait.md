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
but nothing else (regardless of whether it implements any other traits).

Since there needs to be a concrete background type,
you can (as of <!-- date-check --> January 2021) express that type
by using the opaque type in a "defining use site".

```rust,ignore
struct Struct;
impl Bar for Struct { /* stuff */ }
fn foo() -> Foo {
    Struct
}
```

Any other "defining use site" needs to produce the exact same type.

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
