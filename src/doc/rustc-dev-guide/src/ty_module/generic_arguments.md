# ADTs and Generic Arguments

The term `ADT` stands for "Algebraic data type", in rust this refers to a struct, enum, or union.

## ADTs Representation

Let's consider the example of a type like `MyStruct<u32>`, where `MyStruct` is defined like so:

```rust,ignore
struct MyStruct<T> { x: u8, y: T }
```

The type `MyStruct<u32>` would be an instance of `TyKind::Adt`:

```rust,ignore
Adt(&'tcx AdtDef, GenericArgs<'tcx>)
//  ------------  ---------------
//  (1)            (2)
//
// (1) represents the `MyStruct` part
// (2) represents the `<u32>`, or "substitutions" / generic arguments
```

There are two parts:

- The [`AdtDef`][adtdef] references the struct/enum/union but without the values for its type
  parameters. In our example, this is the `MyStruct` part *without* the argument `u32`.
  (Note that in the HIR, structs, enums and unions are represented differently, but in `ty::Ty`,
  they are all represented using `TyKind::Adt`.)
- The [`GenericArgs`] is a list of values that are to be substituted
for the generic parameters.  In our example of `MyStruct<u32>`, we would end up with a list like
`[u32]`. We’ll dig more into generics and substitutions in a little bit.

[adtdef]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.AdtDef.html
[`GenericArgs`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.GenericArgs.html

### **`AdtDef` and `DefId`**

For every type defined in the source code, there is a unique `DefId` (see [this
chapter](../hir.md#identifiers-in-the-hir)). This includes ADTs and generics. In the `MyStruct<T>`
definition we gave above, there are two `DefId`s: one for `MyStruct` and one for `T`.  Notice that
the code above does not generate a new `DefId` for `u32` because it is not defined in that code (it
is only referenced).

`AdtDef` is more or less a wrapper around `DefId` with lots of useful helper methods. There is
essentially a one-to-one relationship between `AdtDef` and `DefId`. You can get the `AdtDef` for a
`DefId` with the [`tcx.adt_def(def_id)` query][adtdefq]. `AdtDef`s are all interned, as shown
by the `'tcx` lifetime.

[adtdefq]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.adt_def

## Question: Why not substitute “inside” the `AdtDef`?

Recall that we represent a generic struct with `(AdtDef, args)`. So why bother with this scheme?

Well, the alternate way we could have chosen to represent types would be to always create a new,
fully-substituted form of the `AdtDef` where all the types are already substituted. This seems like
less of a hassle. However, the `(AdtDef, args)` scheme has some advantages over this.

First, `(AdtDef, args)` scheme has an efficiency win:

```rust,ignore
struct MyStruct<T> {
  ... 100s of fields ...
}

// Want to do: MyStruct<A> ==> MyStruct<B>
```

in an example like this, we can instantiate `MyStruct<A>` as `MyStruct<B>` (and so on) very cheaply,
by just replacing the one reference to `A` with `B`. But if we eagerly instantiated all the fields,
that could be a lot more work because we might have to go through all of the fields in the `AdtDef`
and update all of their types.

A bit more deeply, this corresponds to structs in Rust being [*nominal* types][nominal] — which
means that they are defined by their *name* (and that their contents are then indexed from the
definition of that name, and not carried along “within” the type itself).

[nominal]: https://en.wikipedia.org/wiki/Nominal_type_system


## The `GenericArgs` type

Given a generic type `MyType<A, B, …>`, we have to store the list of generic arguments for `MyType`.

In rustc this is done using [`GenericArgs`]. `GenericArgs` is a thin pointer to a slice of [`GenericArg`] representing a list of generic arguments for a generic item. For example, given a `struct HashMap<K, V>` with two type parameters, `K` and `V`, the `GenericArgs` used to represent the type `HashMap<i32, u32>` would be represented by `&'tcx [tcx.types.i32, tcx.types.u32]`.

`GenericArg` is conceptually an `enum` with three variants, one for type arguments, one for const arguments and one for lifetime arguments.
In practice that is actually represented by [`GenericArgKind`] and [`GenericArg`] is a more space efficient version that has a method to
turn it into a `GenericArgKind`.

The actual `GenericArg` struct stores the type, lifetime or const as an interned pointer with the discriminant stored in the lower 2 bits.
Unless you are working with the `GenericArgs` implementation specifically, you should generally not have to deal with `GenericArg` and instead
make use of the safe [`GenericArgKind`](#genericargkind) abstraction obtainable via the `GenericArg::unpack()` method.

In some cases you may have to construct a `GenericArg`, this can be done via `Ty/Const/Region::into()` or `GenericArgKind::pack`.

```rust,ignore
// An example of unpacking and packing a generic argument.
fn deal_with_generic_arg<'tcx>(generic_arg: GenericArg<'tcx>) -> GenericArg<'tcx> {
    // Unpack a raw `GenericArg` to deal with it safely.
    let new_generic_arg: GenericArgKind<'tcx> = match generic_arg.unpack() {
        GenericArgKind::Type(ty) => { /* ... */ }
        GenericArgKind::Lifetime(lt) => { /* ... */ }
        GenericArgKind::Const(ct) => { /* ... */ }
    };
    // Pack the `GenericArgKind` to store it in a generic args list.
    new_generic_arg.pack()
}
```

[list]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.List.html
[`GenericArg`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.GenericArg.html
[`GenericArgKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.GenericArgKind.html
[`GenericArgs`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.GenericArgs.html

So pulling it all together:

```rust,ignore
struct MyStruct<T>(T);
type Foo = MyStruct<u32>
```

For the `MyStruct<U>` written in the `Foo` type alias, we would represent it in the following way:

- There would be an `AdtDef` (and corresponding `DefId`) for `MyStruct`.
- There would be a `GenericArgs` containing the list `[GenericArgKind::Type(Ty(u32))]`
- And finally a `TyKind::Adt` with the `AdtDef` and `GenericArgs` listed above.
