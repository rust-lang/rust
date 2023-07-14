# Generic arguments
A `ty::GenericArg<'tcx>` represents some entity in the type system: a type
(`Ty<'tcx>`), lifetime (`ty::Region<'tcx>`) or constant (`ty::Const<'tcx>`).
`GenericArg` is used to perform instantiation of generic parameters to concrete
arguments, such as when calling a function with generic parameters explicitly
with type arguments. Instantiations are represented using the
[`GenericArgs` type](#genericargs) as described below.

## `GenericArgs`
`ty::GenericArgs<'tcx>` is intuitively simply a slice of `GenericArg<'tcx>`s,
acting as an ordered list of generic parameters instantiated to
concrete arguments (such as types, lifetimes and consts).

For example, given a `HashMap<K, V>` with two type parameters, `K` and `V`, an
instantiation of the parameters, for example `HashMap<i32, u32>`, would be
represented by `&'tcx [tcx.types.i32, tcx.types.u32]`.

`GenericArgs` provides various convenience methods to instantiate generic arguments
given item definitions, which should generally be used rather than explicitly
instantiating such slices.

## `GenericArg`
The actual `GenericArg` struct is optimised for space, storing the type, lifetime or
const as an interned pointer containing a tag identifying its kind (in the
lowest 2 bits). Unless you are working with the `GenericArgs` implementation
specifically, you should generally not have to deal with `GenericArg` and instead
make use of the safe [`GenericArgKind`](#genericargkind) abstraction.

## `GenericArgKind`
As `GenericArg` itself is not type-safe, the `GenericArgKind` enum provides a more
convenient and safe interface for dealing with generic arguments. An
`GenericArgKind` can be converted to a raw `GenericArg` using `GenericArg::from()`
(or simply `.into()` when the context is clear). As mentioned earlier, instantiation
lists store raw `GenericArg`s, so before dealing with them, it is preferable to
convert them to `GenericArgKind`s first. This is done by calling the `.unpack()`
method.

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
    new_generic_arg.into()
}
```
