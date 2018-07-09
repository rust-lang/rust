# The MIR (Mid-level IR)

MIR is Rust's _Mid-level Intermediate Representation_. It is
constructed from [HIR](./hir.html). MIR was introduced in
[RFC 1211]. It is a radically simplified form of Rust that is used for
certain flow-sensitive safety checks – notably the borrow checker! –
and also for optimization and code generation.

If you'd like a very high-level introduction to MIR, as well as some
of the compiler concepts that it relies on (such as control-flow
graphs and desugaring), you may enjoy the
[rust-lang blog post that introduced MIR][blog].

[blog]: https://blog.rust-lang.org/2016/04/19/MIR.html

## Introduction to MIR

MIR is defined in the [`src/librustc/mir/`][mir] module, but much of the code
that manipulates it is found in [`src/librustc_mir`][mirmanip].

[RFC 1211]: http://rust-lang.github.io/rfcs/1211-mir.html

Some of the key characteristics of MIR are:

- It is based on a [control-flow graph][cfg].
- It does not have nested expressions.
- All types in MIR are fully explicit.

[cfg]: ./appendix/background.html#cfg

## Key MIR vocabulary

This section introduces the key concepts of MIR, summarized here:

- **Basic blocks**: units of the control-flow graph, consisting of:
  - **statements:** actions with one successor
  - **terminators:** actions with potentially multiple successors; always at
    the end of a block
  - (if you're not familiar with the term *basic block*, see the [background
    chapter][cfg])
- **Locals:** Memory locations alloated on the stack (conceptually, at
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

You can get a feeling for how MIR is structed by translating simple
programs into MIR and reading the pretty printed output. In fact, the
playground makes this easy, since it supplies a MIR button that will
show you the MIR for your program. Try putting this program into play
(or [clicking on this link][sample-play]), and then clicking the "MIR"
button on the top:

[sample-play]: https://play.rust-lang.org/?gist=30074856e62e74e91f06abd19bd72ece&version=stable

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
scope 1 {
    let mut _1: std::vec::Vec<i32>;  // "vec" in scope 1 at src/main.rs:2:9: 2:16
}
scope 2 {
}
let mut _2: ();
let mut _3: &mut std::vec::Vec<i32>;
let mut _4: ();
let mut _5: &mut std::vec::Vec<i32>;
```

You can see that variables in MIR don't have names, they have indices,
like `_0` or `_1`.  We also intermingle the user's variables (e.g.,
`_1`) with temporary values (e.g., `_2` or `_3`). You can tell the
difference between user-defined variables have a comment that gives
you their original name (`// "vec" in scope 1...`). The "scope" blocks
(e.g., `scope 1 { .. }`) describe the lexical structure of the source
program (which names were in scope when).

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
possible, and hence we list only one succssor block, `bb2`.

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
= a + b + c` in Rust, that would get compile to two statements and a
temporary:

```mir
TMP1 = a + b
x = TMP1 + c
```

([Try it and see][play-abc], though you may want to do release mode to skip
over the overflow checks.)

[play-abc]: https://play.rust-lang.org/?gist=1751196d63b2a71f8208119e59d8a5b6&version=stable

## MIR data types

The MIR data types are defined in the [`src/librustc/mir/`][mir]
module.  Each of the key concepts mentioned in the previous section
maps in a fairly straightforward way to a Rust type.

The main MIR data type is `Mir`. It contains the data for a single
function (along with sub-instances of Mir for "promoted constants",
but [you can read about those below](#promoted)).

- **Basic blocks**: The basic blocks are stored in the field
  `basic_blocks`; this is a vector of `BasicBlockData`
  structures. Nobody ever references a basic block directly: instead,
  we pass around `BasicBlock` values, which are
  [newtype'd] indices into this vector.
- **Statements** are represented by the type `Statement`.
- **Terminators** are represented by the `Terminator`.
- **Locals** are represented by a [newtype'd] index type `Local`. The
  data for a local variable is found in the `Mir` (the `local_decls`
  vector). There is also a special constant `RETURN_PLACE` identifying
  the special "local" representing the return value.
- **Places** are identified by the enum `Place`. There are a few variants:
  - Local variables like `_1`
  - Static variables `FOO`
  - **Projections**, which are fields or other things that "project
    out" from a base place. So e.g. the place `_1.f` is a projection,
    with `f` being the "projection element and `_1` being the base
    path. `*_1` is also a projection, with the `*` being represented
    by the `ProjectionElem::Deref` element.
- **Rvalues** are represented by the enum `Rvalue`.
- **Operands** are represented by the enum `Operand`.

## Representing constants

*to be written*

<a name="promoted"></a>

### Promoted constants

*to be written*


[mir]: https://github.com/rust-lang/rust/tree/master/src/librustc/mir
[mirmanip]: https://github.com/rust-lang/rust/tree/master/src/librustc_mir
[mir]: https://github.com/rust-lang/rust/tree/master/src/librustc/mir
[newtype'd]: appendix/glossary.html
