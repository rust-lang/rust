# MIR construction

The lowering of [HIR] to [MIR] occurs for the following (probably incomplete)
list of items:

* Function and Closure bodies
* Initializers of `static` and `const` items
* Initializers of enum discriminants
* Glue and Shims of any kind
    * Tuple struct initializer functions
    * Drop code (the `Drop::drop` function is not called directly)
    * Drop implementations of types without an explicit `Drop` implementation

The lowering is triggered by calling the `mir_built` query. The entire lowering
code lives in `src/librustc_mir/build`. There is an intermediate representation
between [HIR] and [MIR] called the `HAIR` that is only used during the lowering.
The `HAIR` has datatypes that mirror the [HIR] datatypes, but instead of e.g. `-x`
being a `hair::ExprKind::Neg(hair::Expr)` it is a `hair::ExprKind::Neg(hir::Expr)`.

This shallowness enables the `HAIR` to represent all datatypes that [HIR] has, but
without having to create an in-memory copy of the entire [HIR]. The `HAIR` also
does a few simplifications, e.g. method calls and function calls have been merged
into a single variant.

The lowering creates local variables for every argument as specified in the signature.
Next it creates local variables for every binding specified (e.g. `(a, b): (i32, String)`)
produces 3 bindings, one for the argument, and two for the bindings. Next it generates
field accesses that read the fields from the argument and writes the value to the binding
variable.

With this initialization out of the way, the lowering triggers a recursive call
to a function that generates the MIR for the body (a `Block` expression) and
writes the result into the `RETURN_PLACE`.

## `unpack!` all the things

One important thing of note is the `unpack!` macro, which accompanies all recursive
calls. The macro ensures, that you get the result of the recursive call while updating
the basic block that you are now in. As an example: lowering `a + b` will need to do
three somewhat independent things:

* give you an `Rvalue` referring to the result of the operation
* insert an assertion ensuring that the operation does not overflow
* tell you in which basic block you should write further operations into, because
  the basic block has changed due to the inserted assertion (assertions terminate
  blocks and jump either to a panic block or a newly created block, the latter being
  the one you get back).

The `unpack!` macro will call the recursive function you pass it, return the `Rvalue`
and update the basic block by mutating the basic block variable you pass to it.

## Lowering expressions into the desired MIR

There are essentially four kinds of representations one might want of a value:

* `Place` refers to a (or a part of) preexisting memory location (local, static, promoted)
* `Rvalue` is something that can be assigned to a `Place`
* `Operand` is an argument to e.g. a `+` operation or a function call
* a temporary variable containing a copy of the value

Since we start out with lowering the function body to an `Rvalue` so we can create an
assignment to `RETURN_PLACE`, that `Rvalue` lowering will in turn trigger lowering to
`Operand` for its arguments (if any). `Operand` lowering either produces a `const`
operand, or moves/copies out of a `Place`, thus triggering a `Place` lowering. An
expression being lowered to a `Place` can in turn trigger a temporary to be created
if the expression being lowered contains operations. This is where the snake bites its
own tail and we need to trigger an `Rvalue` lowering for the expression to be written
into the local.

## Operator lowering

Operators on builtin types are not lowered to function calls (which would end up being
infinite recursion calls, because the trait impls just contain the operation itself
again). Instead there are `Rvalue`s for binary and unary operators and index operations.
These `Rvalue`s later get codegened to llvm primitive operations or llvm intrinsics.

Operators on all other types get lowered to a function call to their `impl` of the
Operator's corresponding trait.

Irrelevant of the lowering kind, the arguments to the operator are lowered to `Operand`s.
This means all arguments are either constants, or refer to an already existing value
somewhere in a local or static.

## Method call lowering

Method calls are lowered to the same `TerminatorKind` that function calls are.
In [MIR] there is no difference between method calls and function calls anymore.

## Conditions

`if` conditions and `match` statements for `enum`s without variants with fields are
lowered to `TerminatorKind::SwitchInt`. Each possible value (so `0` and `1` for `if`
conditions) has a corresponding `BasicBlock` to which the code continues.
The argument being branched on is again an `Operand`.

### Pattern matching

`match` statements for `enum`s with variants that have fields are lowered to
`TerminatorKind::SwitchInt`, too, but the `Operand` refers to a `Place` where the
discriminant of the value can be found. This often involves reading the discriminant
to a new temporary variable.

## Aggregate construction

Aggregate values of any kind are built via `Rvalue::Aggregate`. All fields are
lowered to `Operator`s. This is essentially equivalent to one assignment
statement per aggregate field plus an assignment to the discriminant in the
case of `enum`s.

[MIR]: ./index.html
[HIR]: ../hir.html