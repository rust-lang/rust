# `Binder` and Higher ranked regions

Sometimes we define generic parameters not on an item but as part of a type or a where clause. As an example the type `for<'a> fn(&'a u32)` or the where clause `for<'a> T: Trait<'a>` both introduce a generic lifetime named `'a`. Currently there is no stable syntax for `for<T>` or `for<const N: usize>` but on nightly `feature(non_lifetime_binders)` can be used to write where clauses (but not types) using `for<T>`/`for<const N: usize>`.

The `for` is referred to as a "binder" because it brings new names into scope. In rustc we use the `Binder` type to track where these parameters are introduced and what the parameters are (i.e. how many and whether the parameter is a type/const/region). A type such as `for<'a> fn(&'a u32)` would be
represented in rustc as:
```
Binder(
    fn(&RegionKind::Bound(DebruijnIndex(0), BoundVar(0)) u32) -> (),
    &[BoundVariableKind::Region(...)],
)
```

Usages of these parameters is represented by the `RegionKind::Bound` (or `TyKind::Bound`/`ConstKind::Bound` variants). These bound regions/types/consts are composed of two main pieces of data:
- A [DebruijnIndex](../appendix/background.md#what-is-a-de-bruijn-index) to specify which binder we are referring to.
- A [`BoundVar`] which specifies which of the parameters that the `Binder` introduces we are referring to.

We also sometimes store some extra information for diagnostics reasons via the [`BoundTyKind`]/[`BoundRegionKind`] but this is not important for type equality or more generally the semantics of `Ty`. (omitted from the above example)

In debug output (and also informally when talking to each other) we tend to write these bound variables in the format of `^DebruijnIndex_BoundVar`. The above example would instead be written as `Binder(fn(&'^0_0), &[BoundVariableKind::Region])`. Sometimes when the `DebruijnIndex` is `0` we just omit it and would write `^0`.

Another concrete example, this time a mixture of `for<'a>` in a where clause and a type:
```
where
    for<'a> Foo<for<'b> fn(&'a &'b T)>: Trait,
```
This would be represented as
```
Binder(
    Foo<Binder(
        fn(&'^1_0 &'^0 T/#0),
        [BoundVariableKind::Region(...)]
    )>: Trait,
    [BoundVariableKind::Region(...)]
)
```

Note how the `'^1_0` refers to the `'a` parameter. We use a `DebruijnIndex` of `1` to refer to the binder one level up from the innermost one, and a var of `0` to refer to the first parameter bound which is `'a`. We also use `'^0` to refer to the `'b` parameter, the `DebruijnIndex` is `0` (referring to the innermost binder) so we omit it, leaving only the boundvar of `0` referring to the first parameter bound which is `'b`.

We did not always explicitly track the set of bound vars introduced by each `Binder`, this caused a number of bugs (read: ICEs [#81193](https://github.com/rust-lang/rust/issues/81193), [#79949](https://github.com/rust-lang/rust/issues/79949), [#83017](https://github.com/rust-lang/rust/issues/83017)). By tracking these explicitly we can assert when constructing higher ranked where clauses/types that there are no escaping bound variables or variables from a different binder. See the following example of an invalid type inside of a binder:
```
Binder(
    fn(&'^1_0 &'^1 T/#0),
    &[BoundVariableKind::Region(...)],
)
```
This would cause all kinds of issues as the region `'^1_0` refers to a binder at a higher level than the outermost binder i.e. it is an escaping bound var. The `'^1` region (also writeable as `'^0_1`) is also ill formed as the binder it refers to does not introduce a second parameter. Modern day rustc will ICE when constructing this binder due to both of those reasons, in the past we would have simply allowed this to work and then ran into issues in other parts of the codebase. 

[`Binder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Binder.html
[`BoundVar`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.BoundVar.html
[`BoundRegionKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.BoundRegionKind.html
[`BoundTyKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.BoundTyKind.html
