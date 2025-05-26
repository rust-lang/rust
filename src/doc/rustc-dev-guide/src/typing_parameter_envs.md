# Typing/Parameter Environments

<!-- toc -->

## Typing Environments

When interacting with the type system there are a few variables to consider that can affect the results of trait solving. The set of in-scope where clauses, and what phase of the compiler type system operations are being performed in (the [`ParamEnv`][penv] and [`TypingMode`][tmode] structs respectively).

When an environment to perform type system operations in has not yet been created, the [`TypingEnv`][tenv] can be used to bundle all of the external context required into a single type.

Once a context to perform type system operations in has been created (e.g. an [`ObligationCtxt`][ocx] or [`FnCtxt`][fnctxt]) a `TypingEnv` is typically not stored anywhere as only the `TypingMode` is a property of the whole environment, whereas different `ParamEnv`s can be used on a per-goal basis.

[ocx]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/struct.ObligationCtxt.html
[fnctxt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html

## Parameter Environments

### What is a `ParamEnv`

The [`ParamEnv`][penv] is a list of in-scope where-clauses, it typically corresponds to a specific item's where clauses. Some clauses are not explicitly written but are instead implicitly added in the [`predicates_of`][predicates_of] query, such as `ConstArgHasType` or (some) implied bounds.

In most cases `ParamEnv`s are initially created via the [`param_env` query][query] which returns a `ParamEnv` derived from the provided item's where clauses. A `ParamEnv` can also be created with arbitrary sets of clauses that are not derived from a specific item, such as in [`compare_method_predicate_entailment`][method_pred_entailment] where we create a hybrid `ParamEnv` consisting of the impl's where clauses and the trait definition's function's where clauses.

---

If we have a function such as:
```rust
// `foo` would have a `ParamEnv` of:
// `[T: Sized, T: Trait, <T as Trait>::Assoc: Clone]`
fn foo<T: Trait>()
where
    <T as Trait>::Assoc: Clone,
{}
```
If we were conceptually inside of `foo` (for example, type-checking or linting it) we would use this `ParamEnv` everywhere that we interact with the type system. This would allow things such as [normalization], evaluating generic constants, and proving where clauses/goals, to rely on `T` being sized, implementing `Trait`, etc.

A more concrete example:
```rust
// `foo` would have a `ParamEnv` of:
// `[T: Sized, T: Clone]`
fn foo<T: Clone>(a: T) {
    // when typechecking `foo` we require all the where clauses on `requires_clone`
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
fn foo2<T>(a: T) {
    // When typechecking `foo2` we attempt to prove `T: Clone`.
    // As we are type checking `foo2` we use `foo2`'s environment
    // when trying to prove `T: Clone`.
    //
    // Trying to prove `T: Clone` with a `ParamEnv` of `[T: Sized]` will
    // fail as there is nothing in the environment telling the trait solver
    // that `T` implements `Clone` and there exists no user written impl
    // that could apply.
    requires_clone(a);
}
```

[predicates_of]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/collect/predicates_of/fn.predicates_of.html
[method_pred_entailment]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.compare_method_predicate_entailment.html
[query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.param_env
[normalization]: normalization.md

### Acquiring a `ParamEnv`

Using the wrong [`ParamEnv`][penv] when interacting with the type system can lead to ICEs, illformed programs compiling, or erroring when we shouldn't. See [#82159](https://github.com/rust-lang/rust/pull/82159) and [#82067](https://github.com/rust-lang/rust/pull/82067) as examples of PRs that modified the compiler to use the correct param env and in the process fixed ICEs.

In the large majority of cases, when a `ParamEnv` is required it either already exists somewhere in scope, or above in the call stack and should be passed down. A non exhaustive list of places where you might find an existing `ParamEnv`:
- During typeck `FnCtxt` has a [`param_env` field][fnctxt_param_env]
- When writing late lints the `LateContext` has a [`param_env` field][latectxt_param_env]
- During well formedness checking the `WfCheckingCtxt` has a [`param_env` field][wfckctxt_param_env]
- The `TypeChecker` used for MIR Typeck has a [`param_env` field][mirtypeck_param_env]
- In the next-gen trait solver all `Goal`s have a [`param_env` field][goal_param_env] specifying what environment to prove the goal in
- When editing an existing [`TypeRelation`][typerelation] if it implements [`PredicateEmittingRelation`][predicate_emitting_relation] then a [`param_env` method][typerelation_param_env] will be available.

If you aren't sure if there's a `ParamEnv` in scope somewhere that can be used it can be worth opening a thread in the [`#t-compiler/help`][compiler_help] zulip stream where someone may be able to point out where a `ParamEnv` can be acquired from.

Manually constructing a `ParamEnv` is typically only needed at the start of some kind of top level analysis (e.g. hir typeck or borrow checking). In such cases there are three ways it can be done:
- Calling the [`tcx.param_env(def_id)` query][param_env_query] which returns the environment associated with a given definition.
- Creating an empty environment with [`ParamEnv::empty`][env_empty].
- Using [`ParamEnv::new`][param_env_new] to construct an env with an arbitrary set of where clauses. Then calling [`traits::normalize_param_env_or_error`][normalize_env_or_error] to handle normalizing and elaborating all the where clauses in the env.

Using the `param_env` query is by far the most common way to construct a `ParamEnv` as most of the time the compiler is performing an analysis as part of some specific definition.

Creating an empty environment with `ParamEnv::empty` is typically only done either in codegen (indirectly via [`TypingEnv::fully_monomorphized`][tenv_mono]), or as part of some analysis that do not expect to ever encounter generic parameters (e.g. various parts of coherence/orphan check).

Creating an env from an arbitrary set of where clauses is usually unnecessary and should only be done if the environment you need does not correspond to an actual item in the source code (e.g. [`compare_method_predicate_entailment`][method_pred_entailment]).

[param_env_new]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.new
[normalize_env_or_error]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/fn.normalize_param_env_or_error.html
[fnctxt_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#structfield.param_env
[latectxt_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/context/struct.LateContext.html#structfield.param_env
[wfckctxt_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/wfcheck/struct.WfCheckingCtxt.html#structfield.param_env
[goal_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/canonical/ir/solve/struct.Goal.html#structfield.param_env
[typerelation_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/trait.PredicateEmittingRelation.html#tymethod.param_env
[typerelation]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/relate/trait.TypeRelation.html
[mirtypeck_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/type_check/struct.TypeChecker.html#structfield.param_env
[env_empty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.empty
[param_env_query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#structfield.param_env
[method_pred_entailment]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.compare_method_predicate_entailment.html
[predicate_emitting_relation]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/relate/combine/trait.PredicateEmittingRelation.html
[tenv_mono]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TypingEnv.html#method.fully_monomorphized
[compiler_help]: https://rust-lang.zulipchat.com/#narrow/channel/182449-t-compiler.2Fhelp

### How are `ParamEnv`s constructed

Creating a [`ParamEnv`][pe] is more complicated than simply using the list of where clauses defined on an item as written by the user. We need to both elaborate supertraits into the env and fully normalize all aliases. This logic is handled by [`traits::normalize_param_env_or_error`][normalize_env_or_error] (even though it does not mention anything about elaboration).

#### Elaborating supertraits

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

The `Clone` trait has a `Sized` supertrait however we do not end up with two `T: Sized` bounds in the env (one for the supertrait and one for the implicitly added `T: Sized` bound) as the elaboration process (implemented via [`util::elaborate`][elaborate]) deduplicates where clauses.

A side effect of this is that even if no actual elaboration of supertraits takes place, the existing where clauses in the env are _also_ deduplicated. See the following example:
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
[next-gen-solver]: ./solve/trait-solving.md

#### Normalizing all bounds

In the old trait solver the where clauses stored in `ParamEnv` are required to be fully normalized as otherwise the trait solver will not function correctly. A concrete example of needing to normalize the `ParamEnv` is the following:
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

## Typing Modes

Depending on what context we are performing type system operations in, different behaviour may be required. For example during coherence there are stronger requirements about when we can consider goals to not hold or when we can consider types to be unequal.

Tracking which "phase" of the compiler type system operations are being performed in is done by the [`TypingMode`][tmode] enum. The documentation on the `TypingMode` enum is quite good so instead of repeating it here verbatim we would recommend reading the API documentation directly.

[penv]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[tenv]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TypingEnv.html
[tmode]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TypingMode.html
