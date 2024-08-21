
# How are `ParamEnv`'s constructed internally?

Creating a [`ParamEnv`][pe] is more complicated than simply using the list of where clauses defined on an item as written by the user. We need to both elaborate supertraits into the env and fully normalize all aliases. This logic is handled by [`traits::normalize_param_env_or_error`][normalize_env_or_error] (even though it does not mention anything about elaboration).

## Elaborating supertraits

When we have a function such as `fn foo<T: Copy>()` we would like to be able to prove `T: Clone` inside of the function as the `Copy` trait has a `Clone` supertrait. Constructing a `ParamEnv` looks at all of the trait bounds in the env and explicitly adds new where clauses to the `ParamEnv` for any supertraits found on the traits.

A concrete example would be the following function:
```rust
trait Trait: SuperTrait {}
trait SuperTrait: SuperSuperTrait {}

// `bar`'s unelaborated `ParamEnv` would be:
// `[T: Sized, T: Copy, T: Trait]`
fn bar<T: Copy + Trait>(a: T) {
    requires_impl(a);
}

fn requires_impl<T: Clone + SuperSuperTrait>(a: T) {}
```

If we did not elaborate the env then the `requires_impl` call would fail to typecheck as we would not be able to prove `T: Clone` or `T: SuperSuperTrait`. In practice we elaborate the env which means that `bar`'s `ParamEnv` is actually:
`[T: Sized, T: Copy, T: Clone, T: Trait, T: SuperTrait, T: SuperSuperTrait]`
This allows us to prove `T: Clone` and `T: SuperSuperTrait` when type checking `bar`.

The `Clone` trait has a `Sized` supertrait however we do not end up with two `T: Sized` bounds in the env (one for the supertrait and one for the implicitly added `T: Sized` bound). This is because the elaboration process (implemented via [`util::elaborate`][elaborate]) deduplicates the where clauses to avoid this.

As a side effect this also means that even if no actual elaboration of supertraits takes place, the existing where clauses in the env are _also_ deduplicated. See the following example:
```rust
trait Trait {}
// The unelaborated `ParamEnv` would be:
// `[T: Sized, T: Trait, T: Trait]`
// but after elaboration it would be:
// `[T: Sized, T: Trait]`
fn foo<T: Trait + Trait>() {}
```

The [next-gen trait solver][next-gen-solver] also requires this elaboration to take place.

[elaborate]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/traits/util/fn.elaborate.html
[next-gen-solver]: ../solve/trait-solving.md

## Normalizing all bounds

In the old trait solver the where clauses stored in `ParamEnv` are required to be fully normalized or else the trait solver will not function correctly.  A concrete example of needing to normalize the `ParamEnv` is the following:
```rust
trait Trait<T> {
    type Assoc;
}

trait Other {
    type Bar;
}

impl<T> Other for T {
    type Bar = u32;
}

// `foo`'s unnormalized `ParamEnv` would be:
// `[T: Sized, U: Sized, U: Trait<T::Bar>]`
fn foo<T, U>(a: U) 
where
    U: Trait<<T as Other>::Bar>,
{
    requires_impl(a);
}

fn requires_impl<U: Trait<u32>>(_: U) {}
```

As humans we can tell that `<T as Other>::Bar` is equal to `u32` so the trait bound on `U` is equivalent to `U: Trait<u32>`. In practice trying to prove `U: Trait<u32>` in the old solver in this environment would fail as it is unable to determine that `<T as Other>::Bar` is equal to `u32`.

To work around this we normalize `ParamEnv`'s after constructing them so that `foo`'s `ParamEnv` is actually: `[T: Sized, U: Sized, U: Trait<u32>]` which means the trait solver is now able to use the `U: Trait<u32>` in the `ParamEnv` to determine that the trait bound `U: Trait<u32>` holds.

This workaround does not work in all cases as normalizing associated types requires a `ParamEnv` which introduces a bootstrapping problem. We need a normalized `ParamEnv` in order for normalization to give correct results, but we need to normalize to get that `ParamEnv`. Currently we normalize the `ParamEnv` once using the unnormalized param env and it tends to give okay results in practice even though there are some examples where this breaks ([example]).

In the next-gen trait solver the requirement for all where clauses in the `ParamEnv` to be fully normalized is not present and so we do not normalize when constructing `ParamEnv`s.

[example]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=e6933265ea3e84eaa47019465739992c
[pe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[normalize_env_or_error]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/fn.normalize_param_env_or_error.html
