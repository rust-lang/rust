# Interpreter

The interpreter is a virtual machine for executing MIR without compiling to
machine code. It is usually invoked via `tcx.const_eval_*` functions. The
interpreter is shared between the compiler (for compile-time function
evaluation, CTFE) and the tool [Miri](https://github.com/rust-lang/miri/), which
uses the same virtual machine to detect Undefined Behavior in (unsafe) Rust
code.

If you start out with a constant:

```rust
const FOO: usize = 1 << 12;
```

rustc doesn't actually invoke anything until the constant is either used or
placed into metadata.

Once you have a use-site like:

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
Foo::{{constant}}#0: usize = {
    let mut _0: usize;
    let mut _1: (usize, bool);

    bb0: {
        _1 = CheckedSub(const FOO, const 42usize);
        assert(!move (_1.1: bool), "attempt to subtract with overflow") -> bb1;
    }

    bb1: {
        _0 = move (_1.0: usize);
        return;
    }
}
```

Before the evaluation, a virtual memory location (in this case essentially a
`vec![u8; 4]` or `vec![u8; 8]`) is created for storing the evaluation result.

At the start of the evaluation, `_0` and `_1` are
`Operand::Immediate(Immediate::Scalar(ScalarMaybeUndef::Undef))`. This is quite
a mouthful: [`Operand`] can represent either data stored somewhere in the
[interpreter memory](#memory) (`Operand::Indirect`), or (as an optimization)
immediate data stored in-line.  And [`Immediate`] can either be a single
(potentially uninitialized) [scalar value][`Scalar`] (integer or thin pointer),
or a pair of two of them. In our case, the single scalar value is *not* (yet)
initialized.

When the initialization of `_1` is invoked, the value of the `FOO` constant is
required, and triggers another call to `tcx.const_eval_*`, which will not be shown
here. If the evaluation of FOO is successful, `42` will be subtracted from its
value `4096` and the result stored in `_1` as
`Operand::Immediate(Immediate::ScalarPair(Scalar::Raw { data: 4054, .. },
Scalar::Raw { data: 0, .. })`. The first part of the pair is the computed value,
the second part is a bool that's true if an overflow happened. A `Scalar::Raw`
also stores the size (in bytes) of this scalar value; we are eliding that here.

The next statement asserts that said boolean is `0`. In case the assertion
fails, its error message is used for reporting a compile-time error.

Since it does not fail, `Operand::Immediate(Immediate::Scalar(Scalar::Raw {
data: 4054, .. }))` is stored in the virtual memory it was allocated before the
evaluation. `_0` always refers to that location directly.

After the evaluation is done, the return value is converted from [`Operand`] to
[`ConstValue`] by [`op_to_const`]: the former representation is geared towards
what is needed *during* const evaluation, while [`ConstValue`] is shaped by the
needs of the remaining parts of the compiler that consume the results of const
evaluation.  As part of this conversion, for types with scalar values, even if
the resulting [`Operand`] is `Indirect`, it will return an immediate
`ConstValue::Scalar(computed_value)` (instead of the usual `ConstValue::ByRef`).
This makes using the result much more efficient and also more convenient, as no
further queries need to be executed in order to get at something as simple as a
`usize`.

Future evaluations of the same constants will not actually invoke
the interpreter, but just use the cached result.

[`Operand`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/operand/enum.Operand.html
[`Immediate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/enum.Immediate.html
[`ConstValue`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/consts/enum.ConstValue.html
[`Scalar`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/enum.Scalar.html
[`op_to_const`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/const_eval/eval_queries/fn.op_to_const.html

## Datastructures

The interpreter's outside-facing datastructures can be found in
[rustc_middle/src/mir/interpret](https://github.com/rust-lang/rust/blob/master/compiler/rustc_middle/src/mir/interpret).
This is mainly the error enum and the [`ConstValue`] and [`Scalar`] types. A
`ConstValue` can be either `Scalar` (a single `Scalar`, i.e., integer or thin
pointer), `Slice` (to represent byte slices and strings, as needed for pattern
matching) or `ByRef`, which is used for anything else and refers to a virtual
allocation. These allocations can be accessed via the methods on
`tcx.interpret_interner`.  A `Scalar` is either some `Raw` integer or a pointer;
see [the next section](#memory) for more on that.

If you are expecting a numeric result, you can use `eval_usize` (panics on
anything that can't be represented as a `u64`) or `try_eval_usize` which results
in an `Option<u64>` yielding the `Scalar` if possible.

## Memory

To support any kind of pointers, the interpreter needs to have a "virtual memory" that the
pointers can point to.  This is implemented in the [`Memory`] type.  In the
simplest model, every global variable, stack variable and every dynamic
allocation corresponds to an [`Allocation`] in that memory.  (Actually using an
allocation for every MIR stack variable would be very inefficient; that's why we
have `Operand::Immediate` for stack variables that are both small and never have
their address taken.  But that is purely an optimization.)

Such an `Allocation` is basically just a sequence of `u8` storing the value of
each byte in this allocation.  (Plus some extra data, see below.)  Every
`Allocation` has a globally unique `AllocId` assigned in `Memory`.  With that, a
[`Pointer`] consists of a pair of an `AllocId` (indicating the allocation) and
an offset into the allocation (indicating which byte of the allocation the
pointer points to).  It may seem odd that a `Pointer` is not just an integer
address, but remember that during const evaluation, we cannot know at which
actual integer address the allocation will end up -- so we use `AllocId` as
symbolic base addresses, which means we need a separate offset.  (As an aside,
it turns out that pointers at run-time are
[more than just integers, too](https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#pointer-provenance).)

These allocations exist so that references and raw pointers have something to
point to. There is no global linear heap in which things are allocated, but each
allocation (be it for a local variable, a static or a (future) heap allocation)
gets its own little memory with exactly the required size. So if you have a
pointer to an allocation for a local variable `a`, there is no possible (no
matter how unsafe) operation that you can do that would ever change said pointer
to a pointer to a different local variable `b`.
Pointer arithmetic on `a` will only ever change its offset; the `AllocId` stays the same.

This, however, causes a problem when we want to store a `Pointer` into an
`Allocation`: we cannot turn it into a sequence of `u8` of the right length!
`AllocId` and offset together are twice as big as a pointer "seems" to be.  This
is what the `relocation` field of `Allocation` is for: the byte offset of the
`Pointer` gets stored as a bunch of `u8`, while its `AllocId` gets stored
out-of-band.  The two are reassembled when the `Pointer` is read from memory.
The other bit of extra data an `Allocation` needs is `undef_mask` for keeping
track of which of its bytes are initialized.

### Global memory and exotic allocations

`Memory` exists only during evaluation; it gets destroyed when the
final value of the constant is computed.  In case that constant contains any
pointers, those get "interned" and moved to a global "const eval memory" that is
part of `TyCtxt`.  These allocations stay around for the remaining computation
and get serialized into the final output (so that dependent crates can use
them).

Moreover, to also support function pointers, the global memory in `TyCtxt` can
also contain "virtual allocations": instead of an `Allocation`, these contain an
`Instance`.  That allows a `Pointer` to point to either normal data or a
function, which is needed to be able to evaluate casts from function pointers to
raw pointers.

Finally, the [`GlobalAlloc`] type used in the global memory also contains a
variant `Static` that points to a particular `const` or `static` item.  This is
needed to support circular statics, where we need to have a `Pointer` to a
`static` for which we cannot yet have an `Allocation` as we do not know the
bytes of its value.

[`Memory`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/interpret/struct.Memory.html
[`Allocation`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/struct.Allocation.html
[`Pointer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/struct.Pointer.html
[`GlobalAlloc`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/interpret/enum.GlobalAlloc.html

### Pointer values vs Pointer types

One common cause of confusion in the interpreter is that being a pointer *value* and having
a pointer *type* are entirely independent properties.  By "pointer value", we
refer to a `Scalar::Ptr` containing a `Pointer` and thus pointing somewhere into
the interpreter's virtual memory.  This is in contrast to `Scalar::Raw`, which is just some
concrete integer.

However, a variable of pointer or reference *type*, such as `*const T` or `&T`,
does not have to have a pointer *value*: it could be obtained by casting or
transmuting an integer to a pointer. 
And similarly, when casting or transmuting a reference to some
actual allocation to an integer, we end up with a pointer *value*
(`Scalar::Ptr`) at integer *type* (`usize`).  This is a problem because we
cannot meaningfully perform integer operations such as division on pointer
values.

## Interpretation

Although the main entry point to constant evaluation is the `tcx.const_eval_*`
functions, there are additional functions in
[rustc_const_eval/src/const_eval](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/index.html)
that allow accessing the fields of a `ConstValue` (`ByRef` or otherwise). You should
never have to access an `Allocation` directly except for translating it to the
compilation target (at the moment just LLVM).

The interpreter starts by creating a virtual stack frame for the current constant that is
being evaluated. There's essentially no difference between a constant and a
function with no arguments, except that constants do not allow local (named)
variables at the time of writing this guide.

A stack frame is defined by the `Frame` type in
[rustc_const_eval/src/interpret/eval_context.rs](https://github.com/rust-lang/rust/blob/master/compiler/rustc_const_eval/src/interpret/eval_context.rs)
and contains all the local
variables memory (`None` at the start of evaluation). Each frame refers to the
evaluation of either the root constant or subsequent calls to `const fn`. The
evaluation of another constant simply calls `tcx.const_eval_*`, which produce an
entirely new and independent stack frame.

The frames are just a `Vec<Frame>`, there's no way to actually refer to a
`Frame`'s memory even if horrible shenanigans are done via unsafe code. The only
memory that can be referred to are `Allocation`s.

The interpreter now calls the `step` method (in
[rustc_const_eval/src/interpret/step.rs](https://github.com/rust-lang/rust/blob/master/compiler/rustc_const_eval/src/interpret/step.rs)
) until it either returns an error or has no further statements to execute. Each
statement will now initialize or modify the locals or the virtual memory
referred to by a local. This might require evaluating other constants or
statics, which just recursively invokes `tcx.const_eval_*`.
