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

Constant evaluation returns an [`EvalToValTreeResult`] for type system constants or
[`EvalToConstValueResult`] with either the error, or a representation of the constant.

Constants for the type system are encoded in "valtree representation". The `ValTree` datastructure
allows us to represent

* arrays,
* many structs,
* tuples,
* enums and,
* most primitives.

The basic rule for
being permitted in the type system is that every value must be uniquely represented. In other
words: a specific value must only be representable in one specific way. For example: there is only
one way to represent an array of two integers as a `ValTree`:
`ValTree::Branch(&[ValTree::Leaf(first_int), ValTree;:Leaf(second_int)])`.
Even though theoretically a `[u32; 2]` could be encoded in a `u64` and thus just be a
`ValTree::Leaf(bits_of_two_u32)`, that is not a legal construction of `ValTree`
(and is very complex to do, so it is unlikely anyone is tempted to do so).

These rules also mean that some values are not representable. There can be no `union`s in type
level constants, as it is not clear how they should be represented, because their active variant
is unknown. Similarly there is no way to represent raw pointers, as addresses are unknown at
compile-time and thus we cannot make any assumptions about them. References on the other hand
*can* be represented, as equality for references is defined as equality on their value, so we
ignore their address and just look at the backing value. We must make sure that the pointer values
of the references are not observable at compile time. We thus encode `&42` exactly like `42`.
Any conversion from
valtree back to codegen constants must reintroduce an actual indirection. At codegen time the
addresses may be deduplicated between multiple uses or not, entirely depending on arbitrary
optimization choices.

As a consequence, all decoding of `ValTree` must happen by matching on the type first and making
decisions depending on that. The value itself gives no useful information without the type that
belongs to it.

Other constants get represented as [`ConstValue::Scalar`]
or [`ConstValue::Slice`] if possible. This means that the `const_eval_*`
functions cannot be used to create miri-pointers to the evaluated constant.
If you need the value of a constant inside Miri, you need to directly work with
[`const_to_op`].

[`GlobalId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/struct.GlobalId.html
[`ConstValue::Scalar`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/value/enum.ConstValue.html#variant.Scalar
[`ConstValue::Slice`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/value/enum.ConstValue.html#variant.Slice
[`ConstValue::ByRef`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/value/enum.ConstValue.html#variant.ByRef
[`EvalToConstValueResult`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/error/type.EvalToConstValueResult.html
[`EvalToValTreeResult`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/error/type.EvalToValTreeResult.html
[`const_to_op`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.InterpCx.html#method.const_to_op
