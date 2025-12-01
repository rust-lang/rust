# Coercions
 
Coercions are implicit operations which transform a value into a different type. A coercion *site* is a position where a coercion is able to be implicitly performed. There are two kinds of coercion sites: 
- one-to-one
- LUB (Least-Upper-Bound)

```rust
let one_to_one_coercion: &u32 = &mut 8;

let lub_coercion = match my_bool {
    true => &mut 10,
    false => &12,
};
```

See the Reference page on coercions for descriptions of what coercions exist and what expressions are coercion sites: <https://doc.rust-lang.org/reference/type-coercions.html>

## one-to-one coercions

With a one-to-one coercion we coerce from one singular type to a known target type. In the above example this would be the coercion from `&mut u32` to `&u32`.

A one-to-one coercion can be performed by calling [`FnCtxt::coerce`][fnctxt_coerce].

## LUB coercions

With a LUB coercion we coerce a set of source types to some unknown target type. Unlike one-to-one coercions, a LUB coercion *produces* the target type that all of the source types coerce to.

In the above example this would be the LUB coercion of both `&mut i32` and `&i32`, where we produce the target type `&i32`.

The name "LUB coercion" (Least-Upper-Bound coercion) comes from how this coercion takes a set of types and computes the least coerced/subtyped type that both source types are coercable/subtypeable into.

The general process for performing a LUB coercion is as follows:

```rust ignore
// * 1
let mut coerce = CoerceMany::new(intial_lub_ty);
for expr in exprs {
    // * 2
    let expr_ty = fcx.check_expr_with_expectation(expr, expectation);
    coerce.coerce(fcx, &cause, expr, expr_ty);
}
// * 3
let final_ty = coerce.complete(fcx);
```

There are a few key steps here:
1. Creating the [`CoerceMany`][coerce_many] value and picking an initial lub
2. Typechecking each expression and registering its type as part of the LUB coercion
3. Completing the LUB coercion to get the resulting lubbed type

### Step 1

First we create a [`CoerceMany`][coerce_many] value, this stores all of the state required for the LUB coercion. Unlike one-to-one coercions, a LUB coercion isn't a single function call as we want to intermix typechecking with advancing the LUB coercion.

Creating a `CoerceMany` takes some `initial_lub` type. This is different from the *target* of the coercion which is an output of a LUB coercion rather than an input (unlike a one-to-one coercion).

The initial lub ty should be derived from the [`Expectation`][expectation] for whatever expression this LUB coercion is for. It allows for inference constraints from computing the LUB coercion to propagate into the `Expectation`s used for type checking later expressions participating in the LUB coercion.

See the ["unnecessary inference constraints"][unnecessary_inference_constraints] header for some more information about the effects this has.

If there's no `Expectation` to use then some new infer var should be made for the initial lub ty.

### Step 2

Next, for each expression participating in the LUB coercion, we typecheck it then invoke [`CoerceMany::coerce`][coerce_many_coerce] with its type.

In some cases the expression participating in the LUB coercion doesn't actually exist in the HIR. For example when handling an operand-less `break` or `return` expression we need `()` to participate in the LUB coercion.

In these cases the [`CoerceMany::coerce_forced_unit`][coerce_many_coerce_forced_unit] method can be used.

The `CoerceMany::coerce` and `coerce_forced_unit` methods will both emit errors if the new type causes the LUB coercion to be unsatisfiable. In this case the final type of the LUB coercion will be an error type.

### Step 3

Finally once all expressions have been coerced the final type of the LUB coercion can be obtained by calling [`CoerceMany::complete`][coerce_many_complete].

The resulting type of the LUB coercion is meaningfully different from the initial lub type passed in when constructing the [`CoerceMany`][coerce_many]. You should always take the resulting type of the LUB coercion and perform any necessary checks on it.

## Implementation nuances

### Adjustments

When a coerce operation succeeds we record what kind of coercion it was, for example an unsize coercion or an autoderef etc. This is handled as part of the coerce operation by writing a list of *adjustments* into the in-progress [`TypeckResults`][typeck_results].

When building THIR we take the adjustments stored in the `TypeckResults` and make all of the coercion steps explicit. After this point in the compiler there isn't really a notion of coercions, only explicit casts and subtyping in the MIR.

TODO: write and link to an adjustments chapter here

### How does `CoerceMany` work

[`CoerceMany`][coerce_many] works by repeatedly taking the current lub ty and some new source type, and computing a new lub ty which both types can coerce to. The core logic of taking a pair of types and computing some new third type can be found in [`try_find_coercion_lub`][try_find_coercion_lub].

```rust
fn foo() {}
fn bar() {}

let a = match my_bool {
    true => foo,
    true if other_bool => foo,
    false => bar,
}
```

In this example when type checking the `match` expression a LUB coercion is performed. This LUB coercion starts out with an initial lub ty of some inference variable `?x` due to the let statement having no known type.

There are three expressions that participate in this LUB coercion. The first expression of a LUB coercion is special, instead of computing a new type with the existing initial lub ty, we coerce directly from the first expression to the initial lub ty.

1. After type checking `true => foo,` we wind up with the type `FnDef(Foo)`. We then call [`CoerceMany::coerce`][coerce_many_coerce] which will perform a one-to-one coercion of `FnDef(Foo)` to `?x`. This infers `?x=FnDef(Foo)` giving us a new lub ty for the LUB coercion.
2. After type checking `true if other_bool => foo,` we once again wind up with the type `FnDef(Foo)`. We'll then call `CoerceMany::coerce` which will attempt to compute a new lub ty from our previous lub ty (`FnDef(Foo)`) and the type of this expression (`FnDef(Foo)`). This gives us a lub ty of `FnDef(Foo)`.
3. After type checking `false => bar,` we'll wind up with the type `FnDef(Bar)`. We'll then call `CoerceMany::coerce` which will attempt to compute a new lub ty from our previous lub ty (`FnDef(Foo)`) and the type of this expression (`FnDef(Bar)`). In this case we get the type `fn() -> ()` as we choose to coerce both function item types to a function pointer.

This gives us a final type for the LUB coercion of `fn() -> ()`.

### Transitive coercions

[`CoerceMany`][coerce_many]'s algorithm of repeatedly attempting to coerce the currrent target type to the new type currently results in "Transitive Coercions". It's possible for a step in a LUB coercion to coerce an expression, and then a later step to coerce that expression further. 

```rust
struct Foo;

use std::ops::Deref;

impl Deref for Foo {
    type Target = [u8; 2];
    
    fn deref(&self) -> &[u8; 2] {
        &[1; _]
    }
}

fn main() {
    match () {
        _ if true => &Foo,
        _ if true => &[1_u8; 2],
        _ => &[1_u8; 2] as &[u8],
    };
}
```

Here we have a LUB coercion with an initial lub ty of `?x`. In the first step we do a one-to-one coercion of `&Foo` to `?x` (reminder the first step is special).

In the second step we compute a new lub ty from the current lub ty of `&Foo` and the new type of `&[u8; 2]`. This new lub ty would be `&[u8; 2]` by performing a deref coercion of `&Foo` to `&[u8; 2]` on the first expression.

In the third step we compute a new lub ty from the current lub ty of `&[u8; 2]` and the new type of `&[u8]`. This new lub ty would be `&[u8]` by performing an unsizing coercion of `&[u8; 2]` to `&[u8]` on the first two expressions.

Note how the first expression is coerced twice. Once a deref coercion from `&Foo` to `&[u8; 2]`, and then an unsizing coercion from `&[u8; 2]` to `&[u8]`.

The current implementation of transitive coercions is broken, the previous example actually ICEs on stable. While the logic for performing a LUB coercion can produce transitive coercions just fine, the rest of the compiler is not set up to handle them.

One-to-one coercions are also not capable of producing a lot of the kinds of transitive coercions that LUB coercions can. For example if we take the previous example and turn it into a one-to-one coercion we get a compile error:
```rust
struct Foo;

use std::ops::Deref;

impl Deref for Foo {
    type Target = [u8; 2];
    
    fn deref(&self) -> &[u8; 2] {
        &[1; _]
    }
}

fn main() {
    let a: &[u8] = &Foo;
}
```

Here we try to perform a one-to-one coercion from `&Foo` to `&[u8]` which fails as we can only perform a deref coercion *or* an unsizing coercion, we can't compose the two.

### How does `try_find_coercion_lub` work

There are three ways that we can compute a new lub ty for a LUB coercion:
1. Coerce both the current lub ty and the new type to a function pointer
2. Coerce the current lub ty to the new type (or vice versa)
3. Compute a mutual supertype of the current lub ty and the new type

Unfortunately the actual implementation obsfucates this a fair amount. 

Computing a mutual supertype happens implicitly due to reusing the logic for one-to-one coercions which already handles subtyping if coercing fails.

Additionally when trying to coerce both the current lub ty and the new type to function pointers we eagerly try to compute a mutual supertype to avoid unnecessary coercions.

There is likely room for improving the structure of this function to make it more closely align with the conceptual model.

### `use_lub` field in one-to-one coercions

The implementation of one-to-one coercions is reused as part of LUB coercions.

It would be wrong for LUB coercions to use one way subtyping when relating signatures or falling back to subtyping in the case of no coercions being possible. Instead we want to compute a mutual supertype of the two types.

The `use_lub` field on [`Coerce`][coerce_ty] exists to toggle whether to perform normal subtyping (in the case of a one-to-one coercion), or whether to compute a mutual supertype (in the case of a LUB coercion).

### Lubbing

In theory computing a mutual supertype should be as simple as creating some new infer var `?mutual_sup` and then requiring `lub_ty <: ?mutual_sup` and `new_ty <: ?mutual_sup`. In reality LUB coercions use a special [`TypeRelation`][type_relation], [`LatticeOp`][lattice_op].

This is primarily to work around subtyping/generalization for higher ranked types being fairly broken. Unlike normal subtyping, when encountering higher ranked types the lub type relation will switch to invariance.

This enforces that the binders of the higher ranked types are equivalent which avoids the need to pick a "most general" binder, which would be quite difficult to do.

It also avoids the process of computing a mutual supertype being *order dependent*. Given the types `a` and `b`, it may be nice if computing the mutual supertype of `a` and `b` would yield the same result as computing the mutual supertype of `b` and `a`.

The current issues with higher ranked types and subtyping would cause this property to not hold if we were to use the naive method of computing a mutual supertype.

Coercions being turned into explicit MIR operations during MIR building means that the process of computing the final type of a LUB coercion only occurs during HIR typeck. This also means the behaviour of computing a mutual supertype only matters for type inference, and is not soundness relevant.

## Cautionary notes

### Probes

Care should be taken when coercing from inside of a probe as both one-to-one coercions and LUB coercions have side effects that can't be rolled back by a probe.

LUB coercions will emit error when a coercion step fails, this makes it entirely suitable for use inside of probes.

1-to-1 and LUB coercions will both apply *adjustments* to the coerced expressions on success. This means that if inside of a probe and an attempt to coerce succeeds, then the probe must not rollback anything.

It's therefore correct to wrap a [`FnCtxt::coerce`][fnctxt_coerce] call inside of a [`commit_if_ok`][commit_if_ok], but would be wrong to do so if returning `Err` after the coerce call. It would also be wrong to call `FnCtxt::coerce` from within a [`probe`][probe].

[`CoerceMany`][coerce_many] should never be used from within a `probe` or `commit_if_ok`.

### Never-to-Any coercions

Coercing from the never type (`!`) to an inference variable will result in a [`NeverToAny`][never_to_any] coercion with a target type of the inference variable. This is subtly different from *unifying* the inference variable with the never type.

Unifying some infer var `?x` with `!` requires that `?x` actually be *equal* to `!`. However, a `NeverToAny` coercion allows for `?x` to be inferred to any possible type.

This distinction means that in cases where the initial lub ty of a coercion is an inference variable (e.g. there's no [`Expectation`][expectation] to use for the initial lub ty), it's still important to use a coercion instead of subtyping.

See PR [#147834](https://github.com/rust-lang/rust/pull/147834) which fixes a bug where we were incorrectly inferring things to the never type instead of going through a coercion.

### Fallback to subtyping

Even though subtyping is not a coercion, both [`FnCtxt::coerce`][fnctxt_coerce] and [`CoerceMany::coerce`][coerce_many_coerce]/[`coerce_forced_unit`][coerce_many_coerce_forced_unit] are able to succeed due to subtyping.

For one-to-one coercions we will try to enforce the source type is a subtype of the target type. For LUB coercions we will try to compute a type that is a supertype of all the existing types.

For example performing a one-to-one coercion of `?x` to `u32` will fallback to subtyping, inferring `?x eq u32`. This means that when a coercion fails there's no need to attempt subtyping afterwards.

### Unnecessary inference constraints

Using types from [`Expectation`][expectation]s as the initial lub ty can cause infer vars to be constrained by the types of the expressions participating in the LUB coercion. This is not always desirable as these infer vars actually only need to be constrained by the final type of the LUB coercion.

```rust
fn foo<T>(_: T) {}

fn a() {}
fn b() {}

foo::<?x>(match my_bool {
    true => a,
    false => b,
})
```

Here we have a LUB coercion with the first expression being of type `FnDef(a)` and the second expression being of type `FnDef(b)`. If we use `?x` as the initial lub ty of the LUB coercion then we would get the following behaviour:
- expression 1: infer `?x=FnDef(a)`
- expression 2: find a coercion lub between `FnDef(a), FnDef(b)` resulting in `fn() -> ()`
- the final type of the LUB coercion is `fn() -> ()`. equate `?x eq fn() -> ()`, where `?x` actually already has been inferred to `FnDef(a)`, so this is actually equating `FnDef(a) eq fn() -> ()` which does not hold

To avoid some (but not all) of these undesirable inference constraints, if the `Expectation` for the LUB coercion is an inference variable then we won't use it as the initial lub ty. Instead we create a new infer var, for example in the above code snippet we would actually make some new infer var `?y` for the initial lub ty instead of using `?x`.
- expression 1: infer `?y=FnDef(a)`
- expression 2: find a coercion lub between `FnDef(a), FnDef(b)` resulting in `fn() -> ()`
- the final type of the LUB coercion is `fn() -> ()`, infer `?x=fn() -> ()`

See [#140283](https://github.com/rust-lang/rust/pull/140283) for a case where we had undesirable inference constraints caused by not creating a new infer var.

This doesn't avoid unnecessary constraints in *all* cases, only the most common case of having an infer var as our `Expectation`. In theory it would be desirable to avoid these constraints in all cases but it would be quite involved to do so.

[coerce_many]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/coercion/struct.CoerceMany.html
[coerce_many_coerce]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/coercion/struct.CoerceMany.html#method.coerce
[coerce_many_coerce_forced_unit]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/coercion/struct.CoerceMany.html#method.coerce_forced_unit
[coerce_many_complete]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/coercion/struct.CoerceMany.html#method.complete
[try_find_coercion_lub]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#method.try_find_coercion_lub
[expectation]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/expectation/enum.Expectation.html
[unnecessary_inference_constraints]: #unnecessary-inference-constraints
[typeck_results]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TypeckResults.html
[type_relation]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/canonical/ir/relate/trait.TypeRelation.html
[lattice_op]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/relate/lattice/struct.LatticeOp.html
[fnctxt_coerce]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#method.coerce
[coerce_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/coercion/struct.Coerce.html
[commit_if_ok]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.commit_if_ok
[probe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.probe
[never_to_any]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/adjustment/enum.Adjust.html#variant.NeverToAny
