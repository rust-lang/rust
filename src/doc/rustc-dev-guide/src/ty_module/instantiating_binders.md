# Instantiating `Binder`s

Much like [`EarlyBinder`], when accessing the inside of a [`Binder`] we must first discharge it by replacing the bound vars with some other value. This is for much the same reason as with `EarlyBinder`, types referencing parameters introduced by the `Binder` do not make any sense outside of that binder, for example:
```rust,ignore
fn foo<'a>(a: &'a u32) -> &'a u32 {
    a
}
fn bar<T>(a: fn(&u32) -> T) -> T {
    a(&10)
}

fn main() {
    let higher_ranked_fn_ptr = foo as for<'a> fn(&'a u32) -> &'a u32;
    let references_bound_vars = bar(higher_ranked_fn_ptr);
}
```
In this example we are providing an argument of type `for<'a> fn(&'^0 u32) -> &'^0 u32` to `bar`, we do not want to allow `T` to be inferred to the type `&'^0 u32` as it would be rather nonsensical (and likely unsound if we did not happen to ICE, `main` has no idea what `'a` is so how would the borrow checker handle a borrow with lifetime `'a`).

Unlike `EarlyBinder` we typically do not instantiate `Binder` with some concrete set of arguments from the user, i.e. `['b, 'static]` as arguments to a `for<'a1, 'a2> fn(&'a1 u32, &'a2 u32)`. Instead we usually instantiate the binder with inference variables or placeholders.

## Instantiating with inference variables

We instantiate binders with inference variables when we are trying to infer a possible instantiation of the binder, e.g. calling higher ranked function pointers or attempting to use a higher ranked where-clause to prove some bound. For example, given the `higher_ranked_fn_ptr` from the example above, if we were to call it with `&10_u32` we would: 
- Instantiate the binder with infer vars yielding a signature of `fn(&'?0 u32) -> &'?0 u32)`
- Equate the type of the provided argument `&10_u32` (&'static u32) with the type in the signature, `&'?0 u32`, inferring `'?0 = 'static`
- The provided arguments were correct as we were successfully able to unify the types of the provided arguments with the types of the arguments in fn ptr signature

As another example of instantiating with infer vars, given some `for<'a> T: Trait<'a>` where-clause, if we were attempting to prove that `T: Trait<'static>` holds we would:
- Instantiate the binder with infer vars yielding a where clause of `T: Trait<'?0>`
- Equate the goal of `T: Trait<'static>` with the instantiated where clause, inferring `'?0 = 'static`
- The goal holds because we were successfully able to unify `T: Trait<'static>` with `T: Trait<'?0>`

Instantiating binders with inference variables can be accomplished by using the [`instantiate_binder_with_fresh_vars`] method on [`InferCtxt`]. Binders should be instantiated with infer vars when we only care about one specific instantiation of the binder, if instead we wish to reason about all possible instantiations of the binder then placeholders should be used instead.

## Instantiating with placeholders

Placeholders are very similar to `Ty/ConstKind::Param`/`ReEarlyParam`, they represent some unknown type that is only equal to itself. `Ty`/`Const` and `Region` all have a [`Placeholder`] variant that is comprised of a [`Universe`] and a [`BoundVar`]. 

The `Universe` tracks which binder the placeholder originated from, and the `BoundVar` tracks which parameter on said binder that this placeholder corresponds to. Equality of placeholders is determined solely by whether the universes are equal and the `BoundVar`s are equal. See the [chapter on Placeholders and Universes][ch_placeholders_universes] for more information.

When talking with other rustc devs or seeing `Debug` formatted `Ty`/`Const`/`Region`s, `Placeholder` will often be written as `'!UNIVERSE_BOUNDVARS`. For example given some type `for<'a> fn(&'a u32, for<'b> fn(&'b &'a u32))`, after instantiating both binders (assuming the `Universe` in the current `InferCtxt` was `U0` beforehand), the type of `&'b &'a u32` would be represented as `&'!2_0 &!1_0 u32`.

When the universe of the placeholder is `0`, it will be entirely omitted from the debug output, i.e. `!0_2` would be printed as `!2`. This rarely happens in practice though as we increase the universe in the `InferCtxt` when instantiating a binder with placeholders so usually the lowest universe placeholders encounterable are ones in `U1`.

`Binder`s can be instantiated with placeholders via the [`enter_forall`] method on `InferCtxt`. It should be used whenever the compiler should care about any possible instantiation of the binder instead of one concrete instantiation.

Note: in the original example of this chapter it was mentioned that we should not infer a local variable to have type `&'^0 u32`. This code is prevented from compiling via universes (as explained in the linked chapter)

### Why have both `RePlaceholder` and `ReBound`?

You may be wondering why we have both of these variants, afterall the data stored in `Placeholder` is effectively equivalent to that of `ReBound`: something to track which binder, and an index to track which parameter the `Binder` introduced. 

The main reason for this is that `Bound` is a more syntactic representation of bound variables whereas `Placeholder` is a more semantic representation. As a concrete example:
```rust
impl<'a> Other<'a> for &'a u32 { }

impl<T> Trait for T
where
    for<'a> T: Other<'a>,
{ ... }

impl<T> Bar for T
where
    for<'a> &'a T: Trait
{ ... }
```

Given these trait implementations `u32: Bar` should _not_ hold. `&'a u32` only implements `Other<'a>` when the lifetime of the borrow and the lifetime on the trait are equal. However if we only used `ReBound` and did not have placeholders it may be easy to accidentally believe that trait bound does hold. To explain this let's walk through an example of trying to prove `u32: Bar` in a world where rustc did not have placeholders:
- We start by trying to prove `u32: Bar`
- We find the `impl<T> Bar for T` impl, we would wind up instantiating the `EarlyBinder` with `u32` (note: this is not _quite_ accurate as we first instantiate the binder with an inference variable that we then infer to be `u32` but that distinction is not super important here)
- There is a where clause `for<'a> &'^0 T: Trait` on the impl, as we instantiated the early binder with `u32` we actually have to prove `for<'a> &'^0 u32: Trait`
- We find the `impl<T> Trait for T` impl, we would wind up instantiating the `EarlyBinder` with `&'^0 u32`
- There is a where clause `for<'a> T: Other<'^0>`, as we instantiated the early binder with `&'^0 u32` we actually have to prove `for<'a> &'^0 u32: Other<'^0>`
- We find the `impl<'a> Other<'a> for &'a u32` and this impl is enough to prove the bound as the lifetime on the borrow and on the trait are both `'^0`

This end result is incorrect as we had two separate binders introducing their own generic parameters, the trait bound should have ended up as something like `for<'a1, 'a2> &'^1 u32: Other<'^0>` which is _not_ satisfied by the `impl<'a> Other<'a> for &'a u32`.

While in theory we could make this work it would be quite involved and more complex than the current setup, we would have to:
- "rewrite" bound variables to have a higher `DebruijnIndex` whenever instantiating a `Binder`/`EarlyBinder` with a `Bound` ty/const/region 
- When inferring an inference variable to a bound var, if that bound var is from a binder enterred after creating the infer var, we would have to lower the `DebruijnIndex` of the var.
- Separately track what binder an inference variable was created inside of, also what the innermost binder it can name parameters from (currently we only have to track the latter)
- When resolving inference variables rewrite any bound variables according to the current binder depth of the infcx
- Maybe more (while writing this list items kept getting added so it seems naive to think this is exhaustive)

Fundamentally all of this complexity is because `Bound` ty/const/regions have a different representation for a given parameter on a `Binder` depending on how many other `Binder`s there are between the binder introducing the parameter, and its usage. For example given the following code:
```rust
fn foo<T>()
where
    for<'a> T: Trait<'a, for<'b> fn(&'b T, &'a u32)>
{ ... }
```
That where clause would be written as:  
`for<'a> T: Trait<'^0, for<'b> fn(&'^0 T, &'^1_0 u32)>`  
Despite there being two references to the `'a` parameter they are both represented differently: `^0` and `^1_0`, due to the fact that the latter usage is nested under a second `Binder` for the inner function pointer type.

This is in contrast to `Placeholder` ty/const/regions which do not have this limitation due to the fact that `Universe`s are specific to the current `InferCtxt` not the usage site of the parameter.

It is trivially possible to instantiate `EarlyBinder`s and unify inference variables with existing `Placeholder`s as no matter what context the `Placeholder` is in, it will have the same representation. As an example if we were to instantiate the binder on the higher ranked where clause from above, it would be represented like so:  
`T: Trait<'!1_0, for<'b> fn(&'^0 T, &'!1_0 u32)>`  
the `RePlaceholder` representation for both usages of `'a` are the same despite one being underneath another `Binder`.

If we were to then instantiate the binder on the function pointer we would get a type such as:  
`fn(&'!2_0 T, ^'!1_0 u32)`  
the `RePlaceholder` for the `'b` parameter is in a higher universe to track the fact that its binder was instantiated after the binder for `'a`.

## Instantiating with `ReLateParam`

As discussed in [the chapter about representing types][representing-types], `RegionKind` has two variants for representing generic parameters, `ReLateParam` and `ReEarlyParam`.
`ReLateParam` is conceptually a `Placeholder` that is always in the root universe (`U0`). It is used when instantiating late bound parameters of functions/closures while inside of them. Its actual representation is relatively different from both `ReEarlyParam` and `RePlaceholder`:
- A `DefId` for the item that introduced the late bound generic parameter
- A [`BoundRegionKind`] which either specifies the `DefId` of the generic parameter and its name (via a `Symbol`), or that this placeholder is representing the anonymous lifetime of a `Fn`/`FnMut` closure's self borrow. There is also a variant for `BrAnon` but this is not used for `ReLateParam`.

For example, given the following code:
```rust,ignore
impl Trait for Whatever {
    fn foo<'a>(a: &'a u32) -> &'a u32 {
        let b: &'a u32 = a;
        b
    }
}
``` 
the lifetime `'a` in the type `&'a u32` in the function body would be represented as: 
```
ReLateParam(
    {impl#0}::foo,
    BoundRegionKind::BrNamed({impl#0}::foo::'a, "'a")
)
```

In this specific case of referencing late bound generic parameters of a function from inside the body this is done implicitly during `hir_ty_lowering` rather than explicitly when instantiating a `Binder` somewhere. In some cases however, we do explicitly instantiate a `Binder` with `ReLateParam`s.

Generally whenever we have a `Binder` for late bound parameters on a function/closure and we are conceptually inside of the binder already, we use [`liberate_late_bound_regions`] to instantiate it with `ReLateParam`s. That makes this operation the `Binder` equivalent to `EarlyBinder`'s `instantiate_identity`.

As a concrete example, accessing the signature of a function we are type checking will be represented as `EarlyBinder<Binder<FnSig>>`. As we are already "inside" of these binders, we would call `instantiate_identity` followed by `liberate_late_bound_regions`.

[`liberate_late_bound_regions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.liberate_late_bound_regions
[representing-types]: param_ty_const_regions.md
[`BoundRegionKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.BoundRegionKind.html
[`enter_forall`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/infer/struct.InferCtxt.html#method.enter_forall
[ch_placeholders_universes]: ../borrow_check/region_inference/placeholders_and_universes.md
[`instantiate_binder_with_fresh_vars`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/infer/struct.InferCtxt.html#method.instantiate_binder_with_fresh_vars
[`InferCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/infer/struct.InferCtxt.html
[`EarlyBinder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.EarlyBinder.html
[`Binder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.Binder.html
[`Placeholder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Placeholder.html
[`Universe`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.UniverseIndex.html
[`BoundVar`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.BoundVar.html
