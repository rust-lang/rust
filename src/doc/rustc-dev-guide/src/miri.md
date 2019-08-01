# Miri

Miri (**MIR** **I**nterpreter) is a virtual machine for executing MIR without
compiling to machine code. It is usually invoked via `tcx.const_eval`.

If you start out with a constant

```rust
const FOO: usize = 1 << 12;
```

rustc doesn't actually invoke anything until the constant is either used or
placed into metadata.

Once you have a use-site like

```rust,ignore
type Foo = [u8; FOO - 42];
```

The compiler needs to figure out the length of the array before being able to
create items that use the type (locals, constants, function arguments, ...).

To obtain the (in this case empty) parameter environment, one can call
`let param_env = tcx.param_env(length_def_id);`. The `GlobalId` needed is

```rust,ignore
let gid = GlobalId {
    promoted: None,
    instance: Instance::mono(length_def_id),
};
```

Invoking `tcx.const_eval(param_env.and(gid))` will now trigger the creation of
the MIR of the array length expression. The MIR will look something like this:

```mir
const Foo::{{initializer}}: usize = {
    let mut _0: usize;                   // return pointer
    let mut _1: (usize, bool);

    bb0: {
        _1 = CheckedSub(const Unevaluated(FOO, Slice([])), const 42usize);
        assert(!(_1.1: bool), "attempt to subtract with overflow") -> bb1;
    }

    bb1: {
        _0 = (_1.0: usize);
        return;
    }
}
```

Before the evaluation, a virtual memory location (in this case essentially a
`vec![u8; 4]` or `vec![u8; 8]`) is created for storing the evaluation result.

At the start of the evaluation, `_0` and `_1` are
`ConstValue::Scalar(Scalar::Undef)`. When the initialization of `_1` is invoked, the
value of the `FOO` constant is required, and triggers another call to
`tcx.const_eval`, which will not be shown here. If the evaluation of FOO is
successful, 42 will be subtracted by its value `4096` and the result stored in
`_1` as `ConstValue::ScalarPair(Scalar::Bytes(4054), Scalar::Bytes(0))`. The first
part of the pair is the computed value, the second part is a bool that's true if
an overflow happened.

The next statement asserts that said boolean is `0`. In case the assertion
fails, its error message is used for reporting a compile-time error.

Since it does not fail, `ConstValue::Scalar(Scalar::Bytes(4054))` is stored in the
virtual memory was allocated before the evaluation. `_0` always refers to that
location directly.

After the evaluation is done, the virtual memory allocation is interned into the
`TyCtxt`. Future evaluations of the same constants will not actually invoke
miri, but just extract the value from the interned allocation.

The `tcx.const_eval` function has one additional feature: it will not return a
`ByRef(interned_allocation_id)`, but a `Scalar(computed_value)` if possible. This
makes using the result much more convenient, as no further queries need to be
executed in order to get at something as simple as a `usize`.

## Datastructures

Miri's core datastructures can be found in
[librustc/mir/interpret](https://github.com/rust-lang/rust/blob/master/src/librustc/mir/interpret).
This is mainly the error enum and the `ConstValue` and `Scalar` types. A `ConstValue` can
be either `Scalar` (a single `Scalar`), `ScalarPair` (two `Scalar`s, usually fat
pointers or two element tuples) or `ByRef`, which is used for anything else and
refers to a virtual allocation. These allocations can be accessed via the
methods on `tcx.interpret_interner`.

If you are expecting a numeric result, you can use `unwrap_usize` (panics on
anything that can't be representad as a `u64`) or `assert_usize` which results
in an `Option<u128>` yielding the `Scalar` if possible.

## Allocations

A miri allocation is either a byte sequence of the memory or an `Instance` in
the case of function pointers. Byte sequences can additionally contain
relocations that mark a group of bytes as a pointer to another allocation. The
actual bytes at the relocation refer to the offset inside the other allocation.

These allocations exist so that references and raw pointers have something to
point to. There is no global linear heap in which things are allocated, but each
allocation (be it for a local variable, a static or a (future) heap allocation)
gets its own little memory with exactly the required size. So if you have a
pointer to an allocation for a local variable `a`, there is no possible (no
matter how unsafe) operation that you can do that would ever change said pointer
to a pointer to `b`.

## Interpretation

Although the main entry point to constant evaluation is the `tcx.const_eval`
query, there are additional functions in
[librustc_mir/const_eval.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/const_eval/index.html)
that allow accessing the fields of a `ConstValue` (`ByRef` or otherwise). You should
never have to access an `Allocation` directly except for translating it to the
compilation target (at the moment just LLVM).

Miri starts by creating a virtual stack frame for the current constant that is
being evaluated. There's essentially no difference between a constant and a
function with no arguments, except that constants do not allow local (named)
variables at the time of writing this guide.

A stack frame is defined by the `Frame` type in
[librustc_mir/interpret/eval_context.rs](https://github.com/rust-lang/rust/blob/master/src/librustc_mir/interpret/eval_context.rs)
and contains all the local
variables memory (`None` at the start of evaluation). Each frame refers to the
evaluation of either the root constant or subsequent calls to `const fn`. The
evaluation of another constant simply calls `tcx.const_eval`, which produces an
entirely new and independent stack frame.

The frames are just a `Vec<Frame>`, there's no way to actually refer to a
`Frame`'s memory even if horrible shenanigans are done via unsafe code. The only
memory that can be referred to are `Allocation`s.

Miri now calls the `step` method (in
[librustc_mir/interpret/step.rs](https://github.com/rust-lang/rust/blob/master/src/librustc_mir/interpret/step.rs)
) until it either returns an error or has no further statements to execute. Each
statement will now initialize or modify the locals or the virtual memory
referred to by a local. This might require evaluating other constants or
statics, which just recursively invokes `tcx.const_eval`.
