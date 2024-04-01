# The `ParamEnv` type

## What is it

The type system relies on information in the environment in order for it to function correctly. This information is stored in the [`ParamEnv`][pe] type and it is important to use the correct `ParamEnv` when interacting with the type system.

The information represented by `ParamEnv` is a list of in scope where clauses, and a [`Reveal`][reveal]. A `ParamEnv` typically corresponds to a specific item's environment however it can also be created with arbitrary data that is not derived from a specific item. In most cases `ParamEnv`s are initially created via the [`param_env` query][query] which returns a `ParamEnv` derived from the provided item's where clauses.

If we have a function such as:
```rust
// `foo` would have a `ParamEnv` of:
// `[T: Sized, T: Trait, <T as Trait>::Assoc: Clone]`
fn foo<T: Trait>()
where
    <T as Trait>::Assoc: Clone,
{}
```
If we were interacting with the type system from inside of `foo` we would use this `ParamEnv` everywhere that we interact with the type system. This would allow things such as normalization (TODO: write a chapter about normalization and link it), evaluating generic constants, and proving where clauses/goals, to rely on `T` being sized, implementing `Trait`, etc.

A more concrete example:
```rust
// `foo` would have a `ParamEnv` of:
// `[T: Sized, T: Clone]`
fn foo<T: Clone>(a: T) -> (T, T) {
    // when typechecking `foo` we require all the where clauses on `bar`
    // to hold in order for it to be legal to call. This means we have to
    // prove `T: Clone`. As we are type checking `foo` we use `foo`'s
    // environment when trying to check that `T: Clone` holds.
    //
    // Trying to prove `T: Clone` with a `ParamEnv` of `[T: Sized, T: Clone]`
    // will trivially succeed as bound we want to prove is in our environment.
    requires_clone(a);
}
```

Or alternatively an example that would not compile:
```rust
// `foo2` would have a `ParamEnv` of:
// `[T: Sized]`
fn foo2<T>(a: T) -> (T, T) {
    // When typechecking `foo2` we attempt to prove `T: Clone`.
    // As we are type checking `foo2` we use `foo2`'s environment
    // when trying to prove `T: Clone`.
    //
    // Trying to prove `T: Clone` with a `ParamEnv` of `[T: Sized]` will
    // fail as there is nothing in the environment telling the trait solver
    // that `T` implements `Clone`.
    requires_clone(a);
}
```

It's very important to use the correct `ParamEnv` when interacting with the type system as otherwise it can lead to ICEs or things compiling when they shouldn't (or vice versa). See [#82159](https://github.com/rust-lang/rust/pull/82159) and [#82067](https://github.com/rust-lang/rust/pull/82067) as examples of PRs that changed rustc to use the correct param env to avoid ICE.

## Construction

Creating a `ParamEnv` is more complicated than simply using the list of where clauses defined on an item as written by the user. We need to both elaborate supertraits into the env and fully normalize all aliases. This logic is handled by [`traits::normalize_param_env_or_error`][normalize_env_or_error] (even though it does not mention anything about elaboration).

### Elaborating supertraits

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
[next-gen-solver]: https://rustc-dev-guide.rust-lang.org/solve/trait-solving.html

### Normalizing all bounds

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

---

When needing a `ParamEnv` in the compiler there generally three options for obtaining one:
- The correct env is already in scope simply use it (or pass it down the call stack to where you are)
- Call `tcx.param_env(def_id)`
- Use [`ParamEnv::new`][param_env_new] to construct an env with an arbitrary set of where clauses. Then call `traits::normalize_param_env_or_error` which will handle normalizing and elaborating all the where clauses in the env for you.

In the large majority of cases a `ParamEnv` when required already exists somewhere in scope or above in the call stack and should be passed down.

Using the `param_env` query to obtain an env is generally done at the start of some kind of analysis and then passed everywhere that a `ParamEnv` is required. For example the type checker will create a `ParamEnv` for the item it is type checking and then pass it around everywhere.

Creating an env from an arbitrary set of where clauses is usually unnecessary and should only be done if the environment you need does not correspond to an actual item in the source code.

[param_env_new]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.new
[normalize_env_or_error]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/fn.normalize_param_env_or_error.html

## Bundling

A useful API on `ParamEnv` is the [`and`][and] method which allows bundling a value with the `ParamEnv`. The `and` method produces a [`ParamEnvAnd<T>`][pea] making it clearer that using the inner value is intended to be done in that specific environment.

[and]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.and
[pea]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnvAnd.html
[pe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ty_utils/ty/fn.param_env.html
[reveal]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/traits/enum.Reveal.html