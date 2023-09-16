# The MIR (Mid-level IR)

<!-- toc -->

MIR is Rust's _Mid-level Intermediate Representation_. It is
constructed from [HIR](../hir.html). MIR was introduced in
[RFC 1211]. It is a radically simplified form of Rust that is used for
certain flow-sensitive safety checks – notably the borrow checker! –
and also for optimization and code generation.

If you'd like a very high-level introduction to MIR, as well as some
of the compiler concepts that it relies on (such as control-flow
graphs and desugaring), you may enjoy the
[rust-lang blog post that introduced MIR][blog].

[blog]: https://blog.rust-lang.org/2016/04/19/MIR.html

## Introduction to MIR

MIR is defined in the [`compiler/rustc_middle/src/mir/`][mir] module, but much of the code
that manipulates it is found in [`compiler/rustc_mir_build`][mirmanip_build],
[`compiler/rustc_mir_transform`][mirmanip_transform], and
[`compiler/rustc_mir_dataflow`][mirmanip_dataflow].

[RFC 1211]: https://rust-lang.github.io/rfcs/1211-mir.html

Some of the key characteristics of MIR are:

- It is based on a [control-flow graph][cfg].
- It does not have nested expressions.
- All types in MIR are fully explicit.

[cfg]: ../appendix/background.html#cfg

## Key MIR vocabulary

This section introduces the key concepts of MIR, summarized here:

- **Basic blocks**: units of the control-flow graph, consisting of:
  - **statements:** actions with one successor
  - **terminators:** actions with potentially multiple successors; always at
    the end of a block
  - (if you're not familiar with the term *basic block*, see the [background
    chapter][cfg])
- **Locals:** Memory locations allocated on the stack (conceptually, at
  least), such as function arguments, local variables, and
  temporaries. These are identified by an index, written with a
  leading underscore, like `_1`. There is also a special "local"
  (`_0`) allocated to store the return value.
- **Places:** expressions that identify a location in memory, like `_1` or
  `_1.f`.
- **Rvalues:** expressions that produce a value. The "R" stands for
  the fact that these are the "right-hand side" of an assignment.
  - **Operands:** the arguments to an rvalue, which can either be a
    constant (like `22`) or a place (like `_1`).

You can get a feeling for how MIR is constructed by translating simple
programs into MIR and reading the pretty printed output. In fact, the
playground makes this easy, since it supplies a MIR button that will
show you the MIR for your program. Try putting this program into play
(or [clicking on this link][sample-play]), and then clicking the "MIR"
button on the top:

[sample-play]: https://play.rust-lang.org/?gist=30074856e62e74e91f06abd19bd72ece&version=stable
MIR shown by above link is optimized.
Some statements like `StorageLive` are removed in optimization.
This happens because the compiler notices the value is never accessed in the code.
We can use `rustc [filename].rs -Z mir-opt-level=0 --emit mir` to view unoptimized MIR.
This requires the nightly toolchain.


```rust
fn main() {
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
}
```

You should see something like:

```mir
// WARNING: This output format is intended for human consumers only
// and is subject to change without notice. Knock yourself out.
fn main() -> () {
    ...
}
```

This is the MIR format for the `main` function.

**Variable declarations.** If we drill in a bit, we'll see it begins
with a bunch of variable declarations. They look like this:

```mir
let mut _0: ();                      // return place
let mut _1: std::vec::Vec<i32>;      // in scope 0 at src/main.rs:2:9: 2:16
let mut _2: ();
let mut _3: &mut std::vec::Vec<i32>;
let mut _4: ();
let mut _5: &mut std::vec::Vec<i32>;
```

You can see that variables in MIR don't have names, they have indices,
like `_0` or `_1`.  We also intermingle the user's variables (e.g.,
`_1`) with temporary values (e.g., `_2` or `_3`). You can tell apart
user-defined variables because they have debuginfo associated to them (see below).

**User variable debuginfo.** Below the variable declarations, we find the only
hint that `_1` represents a user variable:
```mir
scope 1 {
    debug vec => _1;                 // in scope 1 at src/main.rs:2:9: 2:16
}
```
Each `debug <Name> => <Place>;` annotation describes a named user variable,
and where (i.e. the place) a debugger can find the data of that variable.
Here the mapping is trivial, but optimizations may complicate the place,
or lead to multiple user variables sharing the same place.
Additionally, closure captures are described using the same system, and so
they're complicated even without optimizations, e.g.: `debug x => (*((*_1).0: &T));`.

The "scope" blocks (e.g., `scope 1 { .. }`) describe the lexical structure of
the source program (which names were in scope when), so any part of the program
annotated with `// in scope 0` would be missing `vec`, if you were stepping
through the code in a debugger, for example.

**Basic blocks.** Reading further, we see our first **basic block** (naturally
it may look slightly different when you view it, and I am ignoring some of the
comments):

```mir
bb0: {
    StorageLive(_1);
    _1 = const <std::vec::Vec<T>>::new() -> bb2;
}
```

A basic block is defined by a series of **statements** and a final
**terminator**.  In this case, there is one statement:

```mir
StorageLive(_1);
```

This statement indicates that the variable `_1` is "live", meaning
that it may be used later – this will persist until we encounter a
`StorageDead(_1)` statement, which indicates that the variable `_1` is
done being used. These "storage statements" are used by LLVM to
allocate stack space.

The **terminator** of the block `bb0` is the call to `Vec::new`:

```mir
_1 = const <std::vec::Vec<T>>::new() -> bb2;
```

Terminators are different from statements because they can have more
than one successor – that is, control may flow to different
places. Function calls like the call to `Vec::new` are always
terminators because of the possibility of unwinding, although in the
case of `Vec::new` we are able to see that indeed unwinding is not
possible, and hence we list only one successor block, `bb2`.

If we look ahead to `bb2`, we will see it looks like this:

```mir
bb2: {
    StorageLive(_3);
    _3 = &mut _1;
    _2 = const <std::vec::Vec<T>>::push(move _3, const 1i32) -> [return: bb3, unwind: bb4];
}
```

Here there are two statements: another `StorageLive`, introducing the `_3`
temporary, and then an assignment:

```mir
_3 = &mut _1;
```

Assignments in general have the form:

```text
<Place> = <Rvalue>
```

A place is an expression like `_3`, `_3.f` or `*_3` – it denotes a
location in memory.  An **Rvalue** is an expression that creates a
value: in this case, the rvalue is a mutable borrow expression, which
looks like `&mut <Place>`. So we can kind of define a grammar for
rvalues like so:

```text
<Rvalue>  = & (mut)? <Place>
          | <Operand> + <Operand>
          | <Operand> - <Operand>
          | ...

<Operand> = Constant
          | copy Place
          | move Place
```

As you can see from this grammar, rvalues cannot be nested – they can
only reference places and constants. Moreover, when you use a place,
we indicate whether we are **copying it** (which requires that the
place have a type `T` where `T: Copy`) or **moving it** (which works
for a place of any type). So, for example, if we had the expression `x
= a + b + c` in Rust, that would get compiled to two statements and a
temporary:

```mir
TMP1 = a + b
x = TMP1 + c
```

([Try it and see][play-abc], though you may want to do release mode to skip
over the overflow checks.)

[play-abc]: https://play.rust-lang.org/?gist=1751196d63b2a71f8208119e59d8a5b6&version=stable

## MIR data types

The MIR data types are defined in the [`compiler/rustc_middle/src/mir/`][mir]
module. Each of the key concepts mentioned in the previous section
maps in a fairly straightforward way to a Rust type.

The main MIR data type is [`Body`]. It contains the data for a single
function (along with sub-instances of Mir for "promoted constants",
but [you can read about those below](#promoted)).

- **Basic blocks**: The basic blocks are stored in the field
  [`Body::basic_blocks`][basicblocks]; this is a vector
  of [`BasicBlockData`] structures. Nobody ever references a
  basic block directly: instead, we pass around [`BasicBlock`]
  values, which are [newtype'd] indices into this vector.
- **Statements** are represented by the type [`Statement`].
- **Terminators** are represented by the [`Terminator`].
- **Locals** are represented by a [newtype'd] index type [`Local`].
  The data for a local variable is found in the
  [`Body::local_decls`][localdecls] vector. There is also a special constant
  [`RETURN_PLACE`] identifying the special "local" representing the return value.
- **Places** are identified by the struct [`Place`]. There are a few
  fields:
  - Local variables like `_1`
  - **Projections**, which are fields or other things that "project
    out" from a base place. These are represented by the [newtype'd] type
    [`ProjectionElem`]. So e.g. the place `_1.f` is a projection,
    with `f` being the "projection element" and `_1` being the base
    path. `*_1` is also a projection, with the `*` being represented
    by the [`ProjectionElem::Deref`] element.
- **Rvalues** are represented by the enum [`Rvalue`].
- **Operands** are represented by the enum [`Operand`].

## Representing constants

When code has reached the MIR stage, constants can generally come in two forms:
*MIR constants* ([`mir::Constant`]) and *type system constants* ([`ty::Const`]).
MIR constants are used as operands: in `x + CONST`, `CONST` is a MIR constant;
similarly, in `x + 2`, `2` is a MIR constant. Type system constants are used in
the type system, in particular for array lengths but also for const generics.

Generally, both kinds of constants can be "unevaluated" or "already evaluated".
And unevaluated constant simply stores the `DefId` of what needs to be evaluated
to compute this result. An evaluated constant (a "value") has already been
computed; their representation differs between type system constants and MIR
constants: MIR constants evaluate to a `mir::ConstValue`; type system constants
evaluate to a `ty::ValTree`.

Type system constants have some more variants to support const generics: they
can refer to local const generic parameters, and they are subject to inference.
Furthermore, the `mir::Constant::Ty` variant lets us use an arbitrary type
system constant as a MIR constant; this happens whenever a const generic
parameter is used as an operand.

### MIR constant values

In general, a MIR constant value (`mir::ConstValue`) was computed by evaluating
some constant the user wrote. This [const evaluation](../const-eval.md) produces
a very low-level representation of the result in terms of individual bytes. We
call this an "indirect" constant (`mir::ConstValue::Indirect`) since the value
is stored in-memory.

However, storing everything in-memory would be awfully inefficient. Hence there
are some other variants in `mir::ConstValue` that can represent certain simple
and common values more efficiently. In particular, everything that can be
directly written as a literal in Rust (integers, floats, chars, bools, but also
`"string literals"` and `b"byte string literals"`) has an optimized variant that
avoids the full overhead of the in-memory representation.

### ValTrees

An evaluated type system constant is a "valtree". The `ty::ValTree` datastructure
allows us to represent

* arrays,
* many structs,
* tuples,
* enums and,
* most primitives.

The most important rule for
this representation is that every value must be uniquely represented. In other
words: a specific value must only be representable in one specific way. For example: there is only
one way to represent an array of two integers as a `ValTree`:
`ValTree::Branch(&[ValTree::Leaf(first_int), ValTree::Leaf(second_int)])`.
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
valtree back a to MIR constant value must reintroduce an actual indirection. At codegen time the
addresses may be deduplicated between multiple uses or not, entirely depending on arbitrary
optimization choices.

As a consequence, all decoding of `ValTree` must happen by matching on the type first and making
decisions depending on that. The value itself gives no useful information without the type that
belongs to it.

<a name="promoted"></a>

### Promoted constants

See the const-eval WG's [docs on promotion](https://github.com/rust-lang/const-eval/blob/master/promotion.md).


[mir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/index.html
[mirmanip_build]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/index.html
[mirmanip_transform]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_transform/index.html
[mirmanip_dataflow]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_dataflow/index.html
[`Body`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Body.html
[newtype'd]: ../appendix/glossary.html#newtype
[basicblocks]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Body.html#structfield.basic_blocks
[`BasicBlock`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.BasicBlock.html
[`BasicBlockData`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.BasicBlockData.html
[`Statement`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Statement.html
[`Terminator`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/terminator/struct.Terminator.html
[`Local`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Local.html
[localdecls]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Body.html#structfield.local_decls
[`RETURN_PLACE`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/constant.RETURN_PLACE.html
[`Place`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Place.html
[`ProjectionElem`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.ProjectionElem.html
[`ProjectionElem::Deref`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.ProjectionElem.html#variant.Deref
[`Rvalue`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.Rvalue.html
[`Operand`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.Operand.html
[`mir::Constant`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/struct.Constant.html
[`ty::Const`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Const.html
