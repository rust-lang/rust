// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Documentation for the trans module

This module contains high-level summaries of how the various modules
in trans work. It is a work in progress. For detailed comments,
naturally, you can refer to the individual modules themselves.

## The Expr module

The expr module handles translation of expressions. The most general
translation routine is `trans()`, which will translate an expression
into a datum. `trans_into()` is also available, which will translate
an expression and write the result directly into memory, sometimes
avoiding the need for a temporary stack slot. Finally,
`trans_to_lvalue()` is available if you'd like to ensure that the
result has cleanup scheduled.

Internally, each of these functions dispatches to various other
expression functions depending on the kind of expression. We divide
up expressions into:

- **Datum expressions:** Those that most naturally yield values.
  Examples would be `22`, `~x`, or `a + b` (when not overloaded).
- **DPS expressions:** Those that most naturally write into a location
  in memory. Examples would be `foo()` or `Point { x: 3, y: 4 }`.
- **Statement expressions:** That that do not generate a meaningful
  result. Examples would be `while { ... }` or `return 44`.

## The Datum module

A `Datum` encapsulates the result of evaluating a Rust expression.  It
contains a `ValueRef` indicating the result, a `ty::t` describing
the Rust type, but also a *kind*. The kind indicates whether the datum
has cleanup scheduled (lvalue) or not (rvalue) and -- in the case of
rvalues -- whether or not the value is "by ref" or "by value".

The datum API is designed to try and help you avoid memory errors like
forgetting to arrange cleanup or duplicating a value. The type of the
datum incorporates the kind, and thus reflects whether it has cleanup
scheduled:

- `Datum<Lvalue>` -- by ref, cleanup scheduled
- `Datum<Rvalue>` -- by value or by ref, no cleanup scheduled
- `Datum<Expr>` -- either `Datum<Lvalue>` or `Datum<Rvalue>`

Rvalue and expr datums are noncopyable, and most of the methods on
datums consume the datum itself (with some notable exceptions). This
reflects the fact that datums may represent affine values which ought
to be consumed exactly once, and if you were to try to (for example)
store an affine value multiple times, you would be duplicating it,
which would certainly be a bug.

Some of the datum methods, however, are designed to work only on
copyable values such as ints or pointers. Those methods may borrow the
datum (`&self`) rather than consume it, but they always include
assertions on the type of the value represented to check that this
makes sense. An example is `shallow_copy_and_take()`, which duplicates
a datum value.

Translating an expression always yields a `Datum<Expr>` result, but
the methods `to_[lr]value_datum()` can be used to coerce a
`Datum<Expr>` into a `Datum<Lvalue>` or `Datum<Rvalue>` as
needed. Coercing to an lvalue is fairly common, and generally occurs
whenever it is necessary to inspect a value and pull out its
subcomponents (for example, a match, or indexing expression). Coercing
to an rvalue is more unusual; it occurs when moving values from place
to place, such as in an assignment expression or parameter passing.

### Lvalues in detail

An lvalue datum is one for which cleanup has been scheduled. Lvalue
datums are always located in memory, and thus the `ValueRef` for an
LLVM value is always a pointer to the actual Rust value. This means
that if the Datum has a Rust type of `int`, then the LLVM type of the
`ValueRef` will be `int*` (pointer to int).

Because lvalues already have cleanups scheduled, the memory must be
zeroed to prevent the cleanup from taking place (presuming that the
Rust type needs drop in the first place, otherwise it doesn't
matter). The Datum code automatically performs this zeroing when the
value is stored to a new location, for example.

Lvalues usually result from evaluating lvalue expressions. For
example, evaluating a local variable `x` yields an lvalue, as does a
reference to a field like `x.f` or an index `x[i]`.

Lvalue datums can also arise by *converting* an rvalue into an lvalue.
This is done with the `to_lvalue_datum` method defined on
`Datum<Expr>`. Basically this method just schedules cleanup if the
datum is an rvalue, possibly storing the value into a stack slot first
if needed. Converting rvalues into lvalues occurs in constructs like
`&foo()` or `match foo() { ref x => ... }`, where the user is
implicitly requesting a temporary.

Somewhat surprisingly, not all lvalue expressions yield lvalue datums
when trans'd. Ultimately the reason for this is to micro-optimize
the resulting LLVM. For example, consider the following code:

    fn foo() -> ~int { ... }
    let x = *foo();

The expression `*foo()` is an lvalue, but if you invoke `expr::trans`,
it will return an rvalue datum. See `deref_once` in expr.rs for
more details.

### Rvalues in detail

Rvalues datums are values with no cleanup scheduled. One must be
careful with rvalue datums to ensure that cleanup is properly
arranged, usually by converting to an lvalue datum or by invoking the
`add_clean` method.

### Scratch datums

Sometimes you need some temporary scratch space.  The functions
`[lr]value_scratch_datum()` can be used to get temporary stack
space. As their name suggests, they yield lvalues and rvalues
respectively. That is, the slot from `lvalue_scratch_datum` will have
cleanup arranged, and the slot from `rvalue_scratch_datum` does not.

## The Cleanup module

The cleanup module tracks what values need to be cleaned up as scopes
are exited, either via failure or just normal control flow. The basic
idea is that the function context maintains a stack of cleanup scopes
that are pushed/popped as we traverse the AST tree. There is typically
at least one cleanup scope per AST node; some AST nodes may introduce
additional temporary scopes.

Cleanup items can be scheduled into any of the scopes on the stack.
Typically, when a scope is popped, we will also generate the code for
each of its cleanups at that time. This corresponds to a normal exit
from a block (for example, an expression completing evaluation
successfully without failure). However, it is also possible to pop a
block *without* executing its cleanups; this is typically used to
guard intermediate values that must be cleaned up on failure, but not
if everything goes right. See the section on custom scopes below for
more details.

Cleanup scopes come in three kinds:
- **AST scopes:** each AST node in a function body has a corresponding
  AST scope. We push the AST scope when we start generate code for an AST
  node and pop it once the AST node has been fully generated.
- **Loop scopes:** loops have an additional cleanup scope. Cleanups are
  never scheduled into loop scopes; instead, they are used to record the
  basic blocks that we should branch to when a `continue` or `break` statement
  is encountered.
- **Custom scopes:** custom scopes are typically used to ensure cleanup
  of intermediate values.

### When to schedule cleanup

Although the cleanup system is intended to *feel* fairly declarative,
it's still important to time calls to `schedule_clean()` correctly.
Basically, you should not schedule cleanup for memory until it has
been initialized, because if an unwind should occur before the memory
is fully initialized, then the cleanup will run and try to free or
drop uninitialized memory. If the initialization itself produces
byproducts that need to be freed, then you should use temporary custom
scopes to ensure that those byproducts will get freed on unwind.  For
example, an expression like `~foo()` will first allocate a box in the
heap and then call `foo()` -- if `foo()` should fail, this box needs
to be *shallowly* freed.

### Long-distance jumps

In addition to popping a scope, which corresponds to normal control
flow exiting the scope, we may also *jump out* of a scope into some
earlier scope on the stack. This can occur in response to a `return`,
`break`, or `continue` statement, but also in response to failure. In
any of these cases, we will generate a series of cleanup blocks for
each of the scopes that is exited. So, if the stack contains scopes A
... Z, and we break out of a loop whose corresponding cleanup scope is
X, we would generate cleanup blocks for the cleanups in X, Y, and Z.
After cleanup is done we would branch to the exit point for scope X.
But if failure should occur, we would generate cleanups for all the
scopes from A to Z and then resume the unwind process afterwards.

To avoid generating tons of code, we cache the cleanup blocks that we
create for breaks, returns, unwinds, and other jumps. Whenever a new
cleanup is scheduled, though, we must clear these cached blocks. A
possible improvement would be to keep the cached blocks but simply
generate a new block which performs the additional cleanup and then
branches to the existing cached blocks.

### AST and loop cleanup scopes

AST cleanup scopes are pushed when we begin and end processing an AST
node. They are used to house cleanups related to rvalue temporary that
get referenced (e.g., due to an expression like `&Foo()`). Whenever an
AST scope is popped, we always trans all the cleanups, adding the cleanup
code after the postdominator of the AST node.

AST nodes that represent breakable loops also push a loop scope; the
loop scope never has any actual cleanups, it's just used to point to
the basic blocks where control should flow after a "continue" or
"break" statement. Popping a loop scope never generates code.

### Custom cleanup scopes

Custom cleanup scopes are used for a variety of purposes. The most
common though is to handle temporary byproducts, where cleanup only
needs to occur on failure. The general strategy is to push a custom
cleanup scope, schedule *shallow* cleanups into the custom scope, and
then pop the custom scope (without transing the cleanups) when
execution succeeds normally. This way the cleanups are only trans'd on
unwind, and only up until the point where execution succeeded, at
which time the complete value should be stored in an lvalue or some
other place where normal cleanup applies.

To spell it out, here is an example. Imagine an expression `~expr`.
We would basically:

1. Push a custom cleanup scope C.
2. Allocate the `~` box.
3. Schedule a shallow free in the scope C.
4. Trans `expr` into the box.
5. Pop the scope C.
6. Return the box as an rvalue.

This way, if a failure occurs while transing `expr`, the custom
cleanup scope C is pushed and hence the box will be freed. The trans
code for `expr` itself is responsible for freeing any other byproducts
that may be in play.

*/
