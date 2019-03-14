# Kinds
A `ty::subst::Kind<'tcx>` represents some entity in the type system: a type
(`Ty<'tcx>`), lifetime (`ty::Region<'tcx>`) or constant (`ty::Const<'tcx>`).
`Kind` is used to perform substitutions of generic parameters for concrete
arguments, such as when calling a function with generic parameters explicitly
with type arguments. Substitutions are represented using the
[`Subst` type](#subst) as described below.

## `Subst`
`ty::subst::Subst<'tcx>` is intuitively simply a slice of `Kind<'tcx>`s,
acting as an ordered list of substitutions from generic parameters to
concrete arguments (such as types, lifetimes and consts).

For example, given a `HashMap<K, V>` with two type parameters, `K` and `V`, an
instantiation of the parameters, for example `HashMap<i32, u32>`, would be
represented by the substitution `&'tcx [tcx.types.i32, tcx.types.u32]`.

`Subst` provides various convenience methods to instantiant substitutions
given item definitions, which should generally be used rather than explicitly
constructing such substitution slices.

## `Kind`
The actual `Kind` struct is optimised for space, storing the type, lifetime or
const as an interned pointer containing a tag identifying its kind (in the
lowest 2 bits). Unless you are working with the `Subst` implementation
specifically, you should generally not have to deal with `Kind` and instead
make use of the safe [`UnpackedKind`](#unpackedkind) abstraction.

## `UnpackedKind`
As `Kind` itself is not type-safe, the `UnpackedKind` enum provides a more
convenient and safe interface for dealing with kinds. An `UnpackedKind` can
be converted to a raw `Kind` using `Kind::from()` (or simply `.into()` when
the context is clear). As mentioned earlier, substition lists store raw
`Kind`s, so before dealing with them, it is preferable to convert them to
`UnpackedKind`s first. This is done by calling the `.unpack()` method.

```rust,ignore
// An example of unpacking and packing a kind.
fn deal_with_kind<'tcx>(kind: Kind<'tcx>) -> Kind<'tcx> {
    // Unpack a raw `Kind` to deal with it safely.
    let new_kind: UnpackedKind<'tcx> = match kind.unpack() {
        UnpackedKind::Type(ty) => { /* ... */ }
        UnpackedKind::Lifetime(lt) => { /* ... */ }
        UnpackedKind::Const(ct) => { /* ... */ }
    };
    // Pack the `UnpackedKind` to store it in a substitution list.
    new_kind.into()
}
```
