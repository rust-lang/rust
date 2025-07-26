# Aliases and Normalization

<!-- toc -->

## Aliases

In Rust there are a number of types that are considered equal to some "underlying" type, for example inherent associated types, trait associated types, free type aliases (`type Foo = u32`), and opaque types (`-> impl RPIT`). We consider such types to be "aliases", alias types are represented by the [`TyKind::Alias`][tykind_alias] variant, with the kind of alias tracked by the [`AliasTyKind`][aliaskind] enum.

Normalization is the process of taking these alias types and replacing them with the underlying type that they are equal to. For example given some type alias `type Foo = u32`, normalizing `Foo` would give `u32`.

The concept of an alias is not unique to *types* and the concept also applies to constants/const generics. However, right now in the compiler we don't really treat const aliases as a "first class concept" so this chapter mostly discusses things in the context of types (even though the concepts transfer just fine).

[tykind_alias]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/enum.TyKind.html#variant.Alias
[aliaskind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/enum.AliasTyKind.html

### Rigid, Ambiguous and Unnormalized Aliases

Aliases can either be "rigid", "ambiguous", or simply unnormalized.

We consider types to be rigid if their "shape" isn't going to change, for example `Box` is rigid as no amount of normalization can turn a `Box` into a `u32`, whereas `<vec::IntoIter<u32> as Iterator>::Item` is not rigid as it can be normalized to `u32`.

Aliases are rigid when we will never be able to normalize them further. A concrete example of a *rigid* alias would be `<T as Iterator>::Item` in an environment where there is no `T: Iterator<Item = ...>` bound, only a `T: Iterator` bound:
```rust
fn foo<T: Iterator>() {
    // This alias is *rigid*
    let _: <T as Iterator>::Item;
}

fn bar<T: Iterator<Item = u32>>() {
    // This alias is *not* rigid as it can be normalized to `u32`
    let _: <T as Iterator>::Item;
}
```

When an alias can't yet be normalized but may wind up normalizable in the [current environment](./typing_parameter_envs.md), we consider it to be an "ambiguous" alias. This can occur when an alias contains inference variables which prevent being able to determine how the trait is implemented:
```rust
fn foo<T: Iterator, U: Iterator>() {
    // This alias is considered to be "ambiguous"
    let _: <_ as Iterator>::Item;
}
```

The reason we call them "ambiguous" aliases is because its *ambiguous* whether this is a rigid alias or not.

The source of the `_: Iterator` trait impl is *ambiguous* (i.e. unknown), it could be some `impl Iterator for u32` or it could be some `T: Iterator` trait bound, we don't know yet. Depending on why `_: Iterator` holds the alias could be an unnormalized alias or it could be a rigid alias; it's *ambiguous* what kind of alias this is.

Finally, an alias can just be unnormalized, `<Vec<u32> as IntoIterator>::Iter` is an unnormalized alias as it can already be normalized to `std::vec::IntoIter<u32>`, it just hasn't been done yet.

---

It is worth noting that Free and Inherent aliases cannot be rigid or ambiguous as naming them also implies having resolved the definition of the alias, which specifies the underlying type of the alias.

### Diverging Aliases

An alias is considered to "diverge" if its definition does not specify an underlying non-alias type to normalize to. A concrete example of diverging aliases:
```rust
type Diverges = Diverges;

trait Trait {
    type DivergingAssoc;
}
impl Trait for () {
    type DivergingAssoc = <() as Trait>::DivergingAssoc;
}
```
In this example both `Diverges` and `DivergingAssoc` are "trivial" cases of diverging type aliases where they have been defined as being equal to themselves. There is no underlying type that `Diverges` can ever be normalized to.

We generally try to error when diverging aliases are defined, but this is entirely a "best effort" check. In the previous example the definitions are "simple enough" to be detected and so errors are emitted. However, in more complex cases, or cases where only some instantiations of generic parameters would result in a diverging alias, we don't emit an error:
```rust
trait Trait {
    type DivergingAssoc<U: Trait>;
}
impl<T: ?Sized> Trait for T {
    // This alias always diverges but we don't emit an error because
    // the compiler can't "see" that.
    type DivergingAssoc<U: Trait> = <U as Trait>::DivergingAssoc<U>;
}
```

Ultimately this means that we have no guarantee that aliases in the type system are non-diverging. As aliases may only diverge for some specific generic arguments, it also means that we only know whether an alias diverges once it is fully concrete. This means that codegen/const-evaluation also has to handle diverging aliases:
```rust
trait Trait {
    type Diverges<U: Trait>;
}
impl<T: ?Sized> Trait for T {
    type Diverges<U: Trait> = <U as Trait>::Diverges<U>;
}

fn foo<T: Trait>() {
    let a: T::Diverges<T>;
}

fn main() {
    foo::<()>();
}
```
In this example we only encounter an error from the diverging alias during codegen of `foo::<()>`, if the call to `foo` is removed then no compilation error will be emitted.

### Opaque Types

Opaque types are a relatively special kind of alias, and are covered in their own chapter: [Opaque types](./opaque-types-type-alias-impl-trait.md).

### Const Aliases

Unlike type aliases, const aliases are not represented directly in the type system, instead const aliases are always an anonymous body containing a path expression to a const item. This means that the only "const alias" in the type system is an anonymous unevaluated const body.

As such there is no `ConstKind::Alias(AliasCtKind::Projection/Inherent/Free, _)`, instead we only have `ConstKind::Unevaluated` which is used for representing anonymous constants.

```rust
fn foo<const N: usize>() {}

const FREE_CONST: usize = 1 + 1;

fn bar() {
    foo::<{ FREE_CONST }>();
    // The const arg is represented with some anonymous constant:
    // ```pseudo-rust
    // const ANON: usize = FREE_CONST; 
    // foo::<ConstKind::Unevaluated(DefId(ANON), [])>();
    // ```
}
```

This is likely to change as const generics functionality is improved, for example `feature(associated_const_equality)` and `feature(min_generic_const_args)` both require handling const aliases similarly to types (without an anonymous constant wrapping all const args).

## What is Normalization

### Structural vs Deep normalization

There are two forms of normalization, structural (sometimes called *shallow*) and deep. Structural normalization should be thought of as only normalizing the "outermost" part of a type. On the other hand deep normalization will normalize *all* aliases in a type.

In practice structural normalization can result in more than just the outer layer of the type being normalized, but this behaviour should not be relied upon. Unnormalizable non-rigid aliases making use of bound variables (`for<'a>`) cannot be normalized by either kind of normalization. 

As an example: conceptually, structurally normalizing the type `Vec<<u8 as Identity>::Assoc>` would be a no-op, whereas deeply normalizing would give `Vec<u8>`. In practice even structural normalization would give `Vec<u8>`, though, again, this should not be relied upon.

Changing the alias to use bound variables will result in different behaviour; `Vec<for<'a> fn(<&'a u8 as Identity>::Assoc)>` would result in no change when structurally normalized, but would result in `Vec<for<'a> fn(&'a u8)>` when deeply normalized.

### Core normalization logic

Structurally normalizing aliases is a little bit more nuanced than replacing the alias with whatever it is defined as being equal to in its definition; the result of normalizing an alias should either be a rigid type or an inference variable (which will later be inferred to a rigid type). To accomplish this we do two things:

First, when normalizing an ambiguous alias it is normalized to an inference variable instead of leaving it as-is, this has two main effects: 
- Even though an inference variable is not a rigid type, it will always wind up inferred *to* a rigid type so we ensure that the result of normalization will not need to be normalized again
- Inference variables are used in all cases where a type is non-rigid, allowing the rest of the compiler to not have to deal with *both* ambiguous aliases *and* inference variables 

Secondly, instead of having normalization directly return the type specified in the definition of the alias, we normalize the type first before returning it[^1]. We do this so that normalization is idempotent/callers do not need to run it in a loop.

```rust
#![feature(lazy_type_alias)]

type Foo<T: Iterator> = Bar<T>;
type Bar<T: Iterator> = <T as Iterator>::Item;

fn foo() {
    let a_: Foo<_>;
}
```

In this example:
- Normalizing `Foo<?x>` would result in `Bar<?x>`, except we want to normalize aliases in the type `Foo` is defined as equal to
- Normalizing `Bar<?x>` would result in `<?x as Iterator>::Item`, except, again, we want to normalize aliases in the type `Bar` is defined as equal to
- Normalizing `<?x as Iterator>::Item` results in some new inference variable `?y`, as `<?x as Iterator>::Item` is an ambiguous alias
- The final result is that normalizing `Foo<?x>` results in `?y`

## How to normalize

When interfacing with the type system it will often be the case that it's necessary to request a type be normalized. There are a number of different entry points to the underlying normalization logic and each entry point should only be used in specific parts of the compiler.

<!-- date-check: May 2025 -->
An additional complication is that the compiler is currently undergoing a transition from the old trait solver to the new trait solver.
As part of this transition our approach to normalization in the compiler has changed somewhat significantly, resulting in some normalization entry points being "old solver only" slated for removal in the long-term once the new solver has stabilized.
The transition can be tracked via the [WG-trait-system-refactor](https://github.com/rust-lang/rust/labels/WG-trait-system-refactor) label in Github.

Here is a rough overview of the different entry points to normalization in the compiler:
- `infcx.at.structurally_normalize`
- `infcx.at.(deeply_)?normalize`
- `infcx.query_normalize`
- `tcx.normalize_erasing_regions`
- `traits::normalize_with_depth(_to)`
- `EvalCtxt::structurally_normalize`

### Outside of the trait solver

The [`InferCtxt`][infcx] type exposes the "main" ways to normalize during analysis: [`normalize`][normalize], [`deeply_normalize`][deeply_normalize] and [`structurally_normalize`][structurally_normalize]. These functions are often wrapped and re-exposed on various `InferCtxt` wrapper types, such as [`FnCtxt`][fcx] or [`ObligationCtxt`][ocx] with minor API tweaks to handle some arguments or parts of the return type automatically.

#### Structural `InferCtxt` normalization

[`infcx.at.structurally_normalize`][structurally_normalize] exposes structural normalization that is able to handle inference variables and regions. It should generally be used whenever inspecting the kind of a type.

Inside of HIR Typeck there is a related method of normalization- [`fcx.structurally_resolve`][structurally_resolve], which will error if the type being resolved is an unresolved inference variable. When the new solver is enabled it will also attempt to structurally normalize the type.

Due to this there is a pattern in HIR typeck where a type is first normalized via `normalize` (only normalizing in the old solver), and then `structurally_resolve`'d (only normalizing in the new solver). This pattern should be preferred over calling `structurally_normalize` during HIR typeck as `structurally_resolve` will attempt to make inference progress by evaluating goals whereas `structurally_normalize` does not.

#### Deep `InferCtxt` normalization

##### `infcx.at.(deeply_)?normalize`

There are two ways to deeply normalize with an `InferCtxt`, `normalize` and `deeply_normalize`. The reason for this is that `normalize` is a "legacy" normalization entry point used only by the old solver, whereas `deeply_normalize` is intended to be the long term way to deeply normalize. Both of these methods can handle regions.

When the new solver is stabilized the `infcx.at.normalize` function will be removed and everything will have been migrated to the new deep or structural normalization methods. For this reason the `normalize` function is a no-op under the new solver, making it suitable only when the old solver needs normalization but the new solver does not.

Using `deeply_normalize` will result in errors being emitted when encountering ambiguous aliases[^2] as it is not possible to support normalizing *all* ambiguous aliases to inference variables[^3]. `deeply_normalize` should generally only be used in cases where we do not expect to encounter ambiguous aliases, for example when working with types from item signatures.

##### `infcx.query_normalize`

[`infcx.query_normalize`][query_norm] is very rarely used, it has almost all the same restrictions as `normalize_erasing_regions` (cannot handle inference variables, no diagnostics support) with the main difference being that it retains lifetime information. For this reason `normalize_erasing_regions` is the better choice in almost all circumstances as it is more efficient due to caching lifetime-erased queries.

In practice `query_normalize` is used for normalization in the borrow checker, and elsewhere as a performance optimization over `infcx.normalize`. Once the new solver is stabilized it is expected that `query_normalize` can be removed from the compiler as the new solvers normalization implementation should be performant enough for it to not be a performance regression.

##### `tcx.normalize_erasing_regions`

[`normalize_erasing_regions`][norm_erasing_regions] is generally used by parts of the compiler that are not doing type system analysis. This normalization entry point does not handle inference variables, lifetimes, or any diagnostics. Lints and codegen make heavy use of this entry point as they typically are working with fully inferred aliases that can be assumed to be well formed (or at least, are not responsible for erroring on). 

[query_norm]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/infer/at/struct.At.html#method.query_normalize
[norm_erasing_regions]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.normalize_erasing_regions
[normalize]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/infer/at/struct.At.html#method.normalize
[deeply_normalize]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/normalize/trait.NormalizeExt.html#tymethod.deeply_normalize
[structurally_normalize]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/trait.StructurallyNormalizeExt.html#tymethod.structurally_normalize_ty
[infcx]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/infer/struct.InferCtxt.html
[fcx]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html
[ocx]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/struct.ObligationCtxt.html
[structurally_resolve]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#method.structurally_resolve_type

### Inside of the trait solver

[`traits::normalize_with_depth(_to)`][norm_with_depth] and [`EvalCtxt::structurally_normalize`][eval_ctxt_structural_norm] are only used by the internals of the trait solvers (old and new respectively). It is effectively a raw entry point to the internals of how normalization is implemented by each trait solver. Other normalization entry points cannot be used from within the internals of trait solving as it wouldn't handle goal cycles and recursion depth correctly.

[norm_with_depth]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/normalize/fn.normalize_with_depth.html
[eval_ctxt_structural_norm]:  https://doc.rust-lang.org/nightly/nightly-rustc/rustc_next_trait_solver/solve/struct.EvalCtxt.html#method.structurally_normalize_term

## When/Where to normalize (Old vs New solver)

One of the big changes between the old and new solver is our approach to when we expect aliases to be normalized.

### Old solver

All types are expected to be normalized as soon as possible, so that all types encountered in the type system are either rigid or an inference variable (which will later be inferred to a rigid term). 

As a concrete example: equality of aliases is implemented by assuming they're rigid and recursively equating the generic arguments of the alias.

### New solver

It's expected that all types potentially contain ambiguous or unnormalized aliases. Whenever an operation is performed that requires aliases to be normalized, it's the responsibility of that logic to normalize the alias (this means that matching on `ty.kind()` pretty much always has to structurally normalize first).

As a concrete example: equality of aliases is implemented by a custom goal kind ([`PredicateKind::AliasRelate`][aliasrelate]) so that it can handle normalization of the aliases itself instead of assuming all alias types being equated are rigid.

Despite this approach we still deeply normalize during [writeback][writeback] for performance/simplicity, so that types in the MIR can still be assumed to have been deeply normalized. 

[aliasrelate]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.PredicateKind.html#variant.AliasRelate
[writeback]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/writeback/index.html

---

There were a few main issues with the old solver's approach to normalization that motivated changing things in the new solver:

### Missing normalization calls

It was a frequent occurrence that normalization calls would be missing, resulting in passing unnormalized types to APIs expecting everything to already be normalized. Treating ambiguous or unnormalized aliases as rigid would result in all sorts of weird errors from aliases not being considered equal to one another, or surprising inference guidance from equating unnormalized aliases' generic arguments.

### Normalizing parameter environments

Another problem was that it was not possible to normalize `ParamEnv`s correctly in the old solver as normalization itself would expect a normalized `ParamEnv` in order to give correct results. See the chapter on `ParamEnv`s for more information: [`Typing/ParamEnv`s: Normalizing all bounds](./typing_parameter_envs.md#normalizing-all-bounds)

### Unnormalizable non-rigid aliases in higher ranked types

Given a type such as `for<'a> fn(<?x as Trait<'a>::Assoc>)`, it is not possible to correctly handle this with the old solver's approach to normalization.

If we were to normalize it to `for<'a> fn(?y)` and register a goal to normalize `for<'a> <?x as Trait<'a>>::Assoc -> ?y`, this would result in errors in cases where `<?x as Trait<'a>>::Assoc` normalized to `&'a u32`. The inference variable `?y` would be in a lower [universe] than the placeholders made when instantiating the `for<'a>` binder.

Leaving the alias unnormalized would also be wrong as the old solver expects all aliases to be rigid. This was a soundness bug before the new solver was stabilized in coherence: [relating projection substs is unsound during coherence](https://github.com/rust-lang/rust/issues/102048).

Ultimately this means that it is not always possible to ensure all aliases inside of a value are rigid.

[universe]: borrow_check/region_inference/placeholders_and_universes.md#what-is-a-universe
[deeply_normalize]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/normalize/trait.NormalizeExt.html#tymethod.deeply_normalize

## Handling uses of diverging aliases

Diverging aliases, like ambiguous aliases, are normalized to inference variables. As normalizing diverging aliases results in trait solver cycles, it always results in an error in the old solver. In the new solver it only results in an error if we wind up requiring all goals to hold in the current context. E.g. normalizing diverging aliases during HIR typeck will result in an error in both solvers.

Alias well formedness doesn't require that the alias doesn't diverge[^4], this means that checking an alias is well formed isn't sufficient to cause an error to be emitted for diverging aliases; actually attempting to normalize the alias is required.

Erroring on diverging aliases being a side effect of normalization means that it is very *arbitrary* whether we actually emit an error, it also differs between the old and new solver as we now normalize in less places.

An example of the ad-hoc nature of erroring on diverging aliases causing "problems":
```rust
trait Trait {
    type Diverges<D: Trait>;
}

impl<T> Trait for T {
    type Diverges<D: Trait> = D::Diverges<D>;
}

struct Bar<T: ?Sized = <u8 as Trait>::Diverges<u8>>(Box<T>);
```

In this example a diverging alias is used but we happen to not emit an error as we never explicitly normalize the defaults of generic parameters. If the `?Sized` opt out is removed then an error is emitted because we wind up happening to normalize a `<u8 as Trait>::Diverges<u8>: Sized` goal which as a side effect results in erroring about the diverging alias.

Const aliases differ from type aliases a bit here; well formedness of const aliases requires that they can be successfully evaluated (via [`ConstEvaluatable`][const_evaluatable] goals). This means that simply checking well formedness of const arguments is sufficient to error if they would fail to evaluate. It is somewhat unclear whether it would make sense to adopt this for type aliases too or if const aliases should stop requiring this for well formedness[^5].

[^1]: In the new solver this is done implicitly

[^2]: There is a subtle difference in how ambiguous aliases in binders are handled between old and new solver. In the old solver we fail to error on some ambiguous aliases inside of higher ranked types whereas the new solver correctly errors.

[^3]: Ambiguous aliases inside of binders cannot be normalized to inference variables, this will be covered more later.

[^4]: As checking aliases are non-diverging cannot be done until they are fully concrete, this would either imply that we cant check aliases are well formed before codegen/const-evaluation or that aliases would go from being well-formed to not well-formed after monomorphization.

[^5]: Const aliases certainly wouldn't be *less* sound than type aliases if we stopped doing this

[const_evaluatable]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ClauseKind.html#variant.ConstEvaluatable
