# Constant Evaluation

Constant evaluation is the process of computing values at compile time. For a
specific item (constant/static/array length) this happens after the MIR for the
item is borrow-checked and optimized. In many cases trying to const evaluate an
item will trigger the computation of its MIR for the first time.

Prominent examples are:

* The initializer of a `static`
* Array length
    * needs to be known to reserve stack or heap space
* Enum variant discriminants
    * needs to be known to prevent two variants from having the same
      discriminant
* Patterns
    * need to be known to check for overlapping patterns

Additionally constant evaluation can be used to reduce the workload or binary
size at runtime by precomputing complex operations at compiletime and only
storing the result.

All uses of constant evaluation can either be categorized as "influencing the type system"
(array lengths, enum variant discriminants, const generic parameters), or as solely being
done to precompute expressions to be used at runtime.

Constant evaluation can be done by calling the `const_eval_*` functions of `TyCtxt`.
They're the wrappers of the `const_eval` query.

* `const_eval_global_id_for_typeck` evaluates a constant to a valtree,
  so the result value can be further inspected by the compiler.
* `const_eval_global_id` evaluate a constant to an "opaque blob" containing its final value;
  this is only useful for codegen backends and the CTFE evaluator engine itself.
* `eval_static_initializer` specifically computes the initial values of a static.
  Statics are special; all other functions do not represent statics correctly
  and have thus assertions preventing their use on statics.

The `const_eval_*` functions use a [`ParamEnv`](./param_env.html) of environment
in which the constant is evaluated (e.g. the function within which the constant is used)
and a [`GlobalId`]. The `GlobalId` is made up of an `Instance` referring to a constant
or static or of an `Instance` of a function and an index into the function's `Promoted` table.

Constant evaluation returns an [`EvalToValTreeResult`] for type system constants
or [`EvalToConstValueResult`] with either the error, or a representation of the
evaluated constant: a [valtree](mir/index.md#valtrees) or a [MIR constant
value](mir/index.md#mir-constant-values), respectively.

[`GlobalId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/struct.GlobalId.html
[`EvalToConstValueResult`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/error/type.EvalToConstValueResult.html
[`EvalToValTreeResult`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/error/type.EvalToValTreeResult.html
