# Effects and effect checking

Note: all of this describes the implementation of the unstable `effects` and
`const_trait_impl` features. None of this implementation is usable or visible from
stable Rust.

The implementation of const traits and `~const` bounds is a limited effect system.
It is used to allow trait bounds on `const fn` to be used within the `const fn` for
method calls. Within the function, in order to know whether a method on a trait
bound is `const`, we need to know whether there is a `~const` bound for the trait.
In order to know whether we can instantiate a `~const` bound on a `const fn`, we
need to know whether there is a `const_trait` impl for the type and trait being
used (or whether the `const fn` is used at runtime, then any type implementing the
trait is ok, just like with other bounds).

We perform these checks via a const generic boolean that gets attached to all
`const fn` and `const trait`. The following sections will explain the desugarings
and the way we perform the checks at call sites.

The const generic boolean is inverted to the meaning of `const`. In the compiler
it is called `host`, because it enables "host APIs" like `static` items, network
access, disk access, random numbers and everything else that isn't available in
`const` contexts. So `false` means "const", `true` means "not const" and if it's
a generic parameter, it means "maybe const" (meaning we're in a const fn or const
trait).

## `const fn`

All `const fn` have a `#[rustc_host] const host: bool` generic parameter that is
hidden from users. Any `~const Trait` bounds in the generics list or `where` bounds
of a `const fn` get converted to `Trait<host> + Trait<true>` bounds. The `Trait<true>`
exists so that associated types of the generic param can be used from projections
like `<T as Trait>::Assoc`, because there are no `<T as ~const Trait>` projections for now.

## `#[const_trait] trait`s

The `#[const_trait]` attribute gives the marked trait a `#[rustc_host] const host: bool`
generic parameter. All functions of the trait "inherit" this generic parameter, just like
they have all the regular generic parameters of the trait. Any `~const Trait` super-trait
bounds get desugared to `Trait<host> + Trait<true>` in order to allow using associated
types and consts of the super traits in the trait declaration. This is necessary, because
`<Self as SuperTrait>::Assoc` is always `<Self as SuperTrait<true>>::Assoc` as there is
no `<Self as ~const SuperTrait>` syntax.

## `typeck` performing method and function call checks.

When generic parameters are instantiated for any items, the `host` generic parameter
is always instantiated as an inference variable. This is a special kind of inference var
that is not part of the type or const inference variables, similar to how we have
special inference variables for type variables that we know to be an integer, but not
yet which one. These separate inference variables fall back to `true` at
the end of typeck (in `fallback_effects`) to ensure that `let _ = some_fn_item_name;`
will keep compiling.

All actually used (in function calls, casts, or anywhere else) function items, will
have the `enforce_context_effects` method invoked.
It trivially returns if the function being called has no `host` generic parameter.

In order to error if a non-const function is called in a const context, we have not
yet disabled the const-check logic that happens on MIR, because
`enforce_context_effects` does not yet perform this check.

The function call's `host` parameter is then equated to the context's `host` value,
which almost always trivially succeeds, as it was an inference var. If the inference
var has already been bound (since the function item is invoked twice), the second
invocation checks it against the first.
