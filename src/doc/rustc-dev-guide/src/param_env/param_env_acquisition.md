
# Which `ParamEnv` do I use?

When needing a [`ParamEnv`][pe] in the compiler there are a few options for obtaining one:
- The correct env is already in scope simply use it (or pass it down the call stack to where you are).
- The [`tcx.param_env(def_id)` query][param_env_query]
- Use [`ParamEnv::new`][param_env_new] to construct an env with an arbitrary set of where clauses. Then call [`traits::normalize_param_env_or_error`][normalize_env_or_error] which will handle normalizing and elaborating all the where clauses in the env for you.
- Creating an empty environment via [`ParamEnv::reveal_all`][env_reveal_all] or [`ParamEnv::empty`][env_empty]

In the large majority of cases a `ParamEnv` when required already exists somewhere in scope or above in the call stack and should be passed down. A non exhaustive list of places where you might find an existing `ParamEnv`:
- During typeck `FnCtxt` has a [`param_env` field][fnctxt_param_env]
- When writing late lints the `LateContext` has a [`param_env` field][latectxt_param_env]
- During well formedness checking the `WfCheckingCtxt` has a [`param_env` field][wfckctxt_param_env]
- The `TypeChecker` used by Mir Typeck has a [`param_env` field][mirtypeck_param_env]
- In the next-gen trait solver all `Goal`s have a [`param_env` field][goal_param_env] specifying what environment to prove the goal in
- When editing an existing [`TypeRelation`][typerelation] if it implements `PredicateEmittingRelation` then a [`param_env` method][typerelation_param_env] will be available.

Using the `param_env` query to obtain an env is generally done at the start of some kind of analysis and then passed everywhere that a `ParamEnv` is required. For example the type checker will create a `ParamEnv` for the item it is type checking and then pass it around everywhere.

Creating an env from an arbitrary set of where clauses is usually unnecessary and should only be done if the environment you need does not correspond to an actual item in the source code (i.e. [`compare_method_predicate_entailment`][method_pred_entailment] as mentioned earlier).

Creating an empty environment via `ParamEnv::empty` is almost always wrong. There are very few places where we actually know that the environment should be empty. One of the only places where we do actually know this is after monomorphization, however the `ParamEnv` there should be constructed via `ParamEnv::reveal_all` instead as at this point we should be able to determine the hidden type of opaque types. Codegen/Post-mono is one of the only places that should be using `ParamEnv::reveal_all`.

An additional piece of complexity here is specifying the `Reveal` (see linked docs for explanation of what reveal does) used for the `ParamEnv`. When constructing a param env using the `param_env` query it will have `Reveal::UserFacing`, if `Reveal::All` is desired then the [`tcx.param_env_reveal_all_normalized`][env_reveal_all_normalized] query can be used instead.

The `ParamEnv` type has a method [`ParamEnv::with_reveal_all_normalized`][with_reveal_all] which converts an existing `ParamEnv` into one with `Reveal::All` specified. Where possible the previously mentioned query should be preferred as it is more efficient.

[param_env_new]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.new
[normalize_env_or_error]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/fn.normalize_param_env_or_error.html
[fnctxt_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#structfield.param_env
[latectxt_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/context/struct.LateContext.html#structfield.param_env
[wfckctxt_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/wfcheck/struct.WfCheckingCtxt.html#structfield.param_env
[goal_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/canonical/ir/solve/struct.Goal.html#structfield.param_env
[typerelation_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/trait.PredicateEmittingRelation.html#tymethod.param_env
[typerelation]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/relate/trait.TypeRelation.html
[mirtypeck_param_env]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/type_check/struct.TypeChecker.html#structfield.param_env
[env_reveal_all_normalized]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.param_env_reveal_all_normalized
[with_reveal_all]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.with_reveal_all_normalized
[env_reveal_all]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.reveal_all
[env_empty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html#method.empty
[pe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[param_env_query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#structfield.param_env
[method_pred_entailment]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.compare_method_predicate_entailment.html
