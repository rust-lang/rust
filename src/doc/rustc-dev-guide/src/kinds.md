# Kinds
A `ty::subst::Kind<'tcx>` represents some entity in the type system: currently
either a type (`Ty<'tcx>`) or a lifetime (`ty::Region<'tcx>`), though in the
future this will also include constants (`ty::Const<'tcx>`) to facilitate the
use of const generics. `Kind` is used for type and lifetime substitution (from
abstract type and lifetime parameters to concrete types and lifetimes).

## `UnpackedKind`
As `Kind` itself is not type-safe (see [`Kind`](#kind)), the `UnpackedKind` enum
provides a more convenient and safe interface for dealing with kinds. To
convert from an `UnpackedKind` to a `Kind`, you can call `Kind::from` (or
`.into`). It should not be necessary to convert a `Kind` to an `UnpackedKind`:
instead, you should prefer to deal with `UnpackedKind`, converting it only when
passing it to `Subst` methods.

## `Kind`
The actual `Kind` struct is optimised for space, storing the type or lifetime
as an interned pointer containing a mask identifying its kind (in the lowest
2 bits).

## `Subst`
`ty::subst::Subst<'tcx>` is simply defined as a slice of `Kind<'tcx>`s
and acts as an ordered list of substitutions from kind parameters (i.e.
type and lifetime parameters) to kinds.

For example, given a `HashMap<K, V>` with two type parameters, `K` and `V`, an
instantiation of the parameters, for example `HashMap<i32, u32>`, would be
represented by the substitution `&'tcx [tcx.types.i32, tcx.types.u32]`.

`Subst` provides various convenience methods to instantiant substitutions
given item definitions.
