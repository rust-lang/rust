# Constant Evaluation

Constant evaluation is the process of computing values at compile time. For a
specific item (constant/static/array length) this happens after the MIR for the
item is borrow-checked and optimized. In many cases trying to const evaluate an
item will trigger the computation of its MIR for the first time.

Prominent examples are

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

Constant evaluation can be done by calling the `const_eval` query of `TyCtxt`.

The `const_eval` query takes a [`ParamEnv`](./param_env.html) of environment in
which the constant is evaluated (e.g. the function within which the constant is
used) and a `GlobalId`. The `GlobalId` is made up of an
`Instance` referring to a constant or static or of an
`Instance` of a function and an index into the function's `Promoted` table.

Constant evaluation returns a `Result` with either the error, or the simplest
representation of the constant. "simplest" meaning if it is representable as an
integer or fat pointer, it will directly yield the value (via `ConstValue::Scalar` or
`ConstValue::ScalarPair`), instead of referring to the [`miri`](./miri.html) virtual
memory allocation (via `ConstValue::ByRef`). This means that the `const_eval`
function cannot be used to create miri-pointers to the evaluated constant or
static. If you need that, you need to directly work with the functions in
[src/librustc_mir/const_eval.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/const_eval/index.html).
