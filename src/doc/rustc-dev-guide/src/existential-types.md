# Existential Types

Existential types are essentially strong type aliases which only expose
a specific set of traits as their interface and the concrete type in the
background is inferred from a certain set of use sites of the existential
type.

In the language they are expressed via

```rust,ignore
existential type Foo: Bar;
```

This is in existential type named `Foo` which can be interacted with via
the `Bar` trait's interface.

Since there needs to be a concrete background type, you can currently
express that type by using the existential type in a "defining use site".

```rust,ignore
struct Struct;
impl Bar for Struct { /* stuff */ }
fn foo() -> Foo {
    Struct
}
```

Any other "defining use site" needs to produce the exact same type.

## Defining use site(s)

Currently only the return value of a function inside can
be a defining use site of an existential type (and only if the return
type of that function contains the existential type).

The defining use of an existential type can be any code *within* the parent
of the existential type definition. This includes any siblings of the
existential type and all children of the siblings.

The initiative for *"not causing fatal brain damage to developers due to
accidentally running infinite loops in their brain while trying to
comprehend what the type system is doing"* has decided to disallow children
of existential types to be defining use sites.

### Associated existential types

Associated existential types can be defined by any other associated item
on the same trait `impl` or a child of these associated items.
