# Parameter `Ty`/`Const`/`Region`s

When inside of generic items, types can be written that use in scope generic parameters, for example `fn foo<'a, T>(_: &'a Vec<T>)`. In this specific case
the `&'a Vec<T>` type would be represented internally as:
```
TyKind::Ref(
  RegionKind::LateParam(DefId(foo), DefId(foo::'a), "'a"),
  TyKind::Adt(Vec, &[TyKind::Param("T", 0)])
)
```

There are three separate ways we represent usages of generic parameters:
- [`TyKind::Param`]/[`ConstKind::Param`]/[`RegionKind::EarlyParam`] for early bound generic parameters (note: all type and const parameters are considered early bound, see the [chapter on early vs late bound parameters][ch_early_late_bound] for more information)
- [`TyKind::Bound`]/[`ConstKind::Bound`]/[`RegionKind::Bound`] for references to parameters introduced via higher ranked bounds or higher ranked types i.e. `for<'a> fn(&'a u32)` or `for<'a> T: Trait<'a>`. This is discussed in the [chapter on `Binder`s][ch_binders].
- [`RegionKind::LateParam`] for late bound lifetime parameters, `LateParam` is discussed in the [chapter on instantiating `Binder`s][ch_instantiating_binders].

This chapter only covers `TyKind::Param` `ConstKind::Param` and `RegionKind::EarlyParam`.

## Ty/Const Parameters

As `TyKind::Param` and `ConstKind::Param` are implemented identically this section only refers to `TyKind::Param` for simplicity.
However you should keep in mind that everything here also is true of `ConstKind::Param`

Each `TyKind::Param` contains two things: the name of the parameter and an index.

See the following concrete example of a usage of `TyKind::Param`:
```rust,ignore
struct Foo<T>(Vec<T>);
```
The `Vec<T>` type is represented as `TyKind::Adt(Vec, &[GenericArgKind::Type(Param("T", 0))])`.

The name is somewhat self explanatory, it's the name of the type parameter. The index of the type parameter is an integer indicating
its order in the list of generic parameters in scope (note: this includes parameters defined on items on outer scopes than the item the parameter is defined on). Consider the following examples:

```rust,ignore
struct Foo<A, B> {
  // A would have index 0
  // B would have index 1

  .. // some fields
}
impl<X, Y> Foo<X, Y> {
  fn method<Z>() {
    // inside here, X, Y and Z are all in scope
    // X has index 0
    // Y has index 1
    // Z has index 2
  }
}
```

Concretely given the `ty::Generics` for the item the parameter is defined on, if the index is `2` then starting from the root `parent`, it will be the third parameter to be introduced. For example in the above example, `Z` has index `2` and is the third generic parameter to be introduced, starting from the `impl` block. 

The index fully defines the `Ty` and is the only part of `TyKind::Param` that matters for reasoning about the code we are compiling. 

Generally we do not care what the name is and only use the index. The name is included for diagnostics and debug logs as otherwise it would be
incredibly difficult to understand the output, i.e. `Vec<Param(0)>: Sized` vs `Vec<T>: Sized`. In debug output, parameter types are
often printed out as `{name}/#{index}`, for example in the function `foo` if we were to debug print `Vec<T>` it would be written as `Vec<T/#0>`.

An alternative representation would be to only have the name, however using an index is more efficient as it means we can index into `GenericArgs` when instantiating generic parameters with some arguments. We would otherwise have to store `GenericArgs` as a `HashMap<Symbol, GenericArg>` and do a hashmap lookup everytime we used a generic item.

In theory an index would also allow for having multiple distinct parameters that use the same name, e.g. 
`impl<A> Foo<A> { fn bar<A>() { .. } }`.
The rules against shadowing make this difficult but those language rules could change in the future.

### Lifetime parameters

In contrast to `Ty`/`Const`'s `Param` singular `Param` variant, lifetimes have two variants for representing region parameters: [`RegionKind::EarlyParam`] and [`RegionKind::LateParam`]. The reason for this is due to function's distinguishing between [early and late bound parameters][ch_early_late_bound] which is discussed in an earlier chapter (see link).

`RegionKind::EarlyParam` is structured identically to `Ty/Const`'s `Param` variant, it is simply a `u32` index and a `Symbol`. For lifetime parameters defined on non-function items we always use `ReEarlyParam`. For functions we use `ReEarlyParam` for any early bound parameters and `ReLateParam` for any late bound parameters. Note that just like `Ty` and `Const` params we often debug format them as `'SYMBOL/#INDEX`, see for example:

```rust,ignore
// This function would have its signature represented as:
//
// ```
// fn(
//     T/#2,
//     Ref('a/#0, Ref(ReLateParam(...), u32))
// ) -> Ref(ReLateParam(...), u32)
// ```
fn foo<'a, 'b, T: 'a>(one: T, two: &'a &'b u32) -> &'b u32 {
    ...
}
```

`RegionKind::LateParam` is discussed more in the chapter on [instantiating binders][ch_instantiating_binders].

[ch_early_late_bound]: ../early_late_parameters.md
[ch_binders]: ./binders.md
[ch_instantiating_binders]: ./instantiating_binders.md
[`BoundRegionKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.BoundRegionKind.html
[`RegionKind::EarlyParam`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.RegionKind.html#variant.ReEarlyParam
[`RegionKind::LateParam`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.RegionKind.html#variant.ReLateParam
[`ConstKind::Param`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html#variant.Param
[`TyKind::Param`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TyKind.html#variant.Param
[`TyKind::Bound`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TyKind.html#variant.Bound
[`ConstKind::Bound`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html#variant.Bound
[`RegionKind::Bound`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.RegionKind.html#variant.ReBound
