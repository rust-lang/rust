# Closure Capture Inference

This section describes how rustc handles closures. Closures in Rust are
effectively "desugared" into structs that contain the values they use (or
references to the values they use) from their creator's stack frame. rustc has
the job of figuring out which values a closure uses and how, so it can decide
whether to capture a given variable by shared reference, mutable reference, or
by move. rustc also has to figure out which of the closure traits ([`Fn`][fn],
[`FnMut`][fn_mut], or [`FnOnce`][fn_once]) a closure is capable of
implementing.

[fn]: https://doc.rust-lang.org/std/ops/trait.Fn.html
[fn_mut]:https://doc.rust-lang.org/std/ops/trait.FnMut.html
[fn_once]: https://doc.rust-lang.org/std/ops/trait.FnOnce.html

Let's start with a few examples:

### Example 1

To start, let's take a look at how the closure in the following example is desugared:

```rust
fn closure(f: impl Fn()) {
    f();
}

fn main() {
    let x: i32 = 10;
    closure(|| println!("Hi {}", x));  // The closure just reads x.
    println!("Value of x after return {}", x);
}
```

Let's say the above is the content of a file called `immut.rs`. If we compile
`immut.rs` using the following command. The [`-Z dump-mir=all`][dump-mir] flag will cause
`rustc` to generate and dump the [MIR][mir] to a directory called `mir_dump`.
```console
> rustc +stage1 immut.rs -Z dump-mir=all
```

[mir]: ./mir/index.md
[dump-mir]: ./mir/passes.md

After we run this command, we will see a newly generated directory in our
current working directory called `mir_dump`, which will contain several files.
If we look at file `rustc.main.-------.mir_map.0.mir`, we will find, among
other things, it also contains this line:

```rust,ignore
_4 = &_1;
_3 = [closure@immut.rs:7:13: 7:36] { x: move _4 };
```

Note that in the MIR examples in this chapter, `_1` is `x`.

Here in first line `_4 = &_1;`, the `mir_dump` tells us that `x` was borrowed
as an immutable reference.  This is what we would hope as our closure just
reads `x`.

### Example 2

Here is another example:

```rust
fn closure(mut f: impl FnMut()) {
    f();
}

fn main() {
    let mut x: i32 = 10;
    closure(|| {
        x += 10;  // The closure mutates the value of x
        println!("Hi {}", x)
    });
    println!("Value of x after return {}", x);
}
```

```rust,ignore
_4 = &mut _1;
_3 = [closure@mut.rs:7:13: 10:6] { x: move _4 };
```
This time along, in the line `_4 = &mut _1;`, we see that the borrow is changed to mutable borrow.
Fair enough! The closure increments `x` by 10.

### Example 3

One more example:

```rust
fn closure(f: impl FnOnce()) {
    f();
}

fn main() {
    let x = vec![21];
    closure(|| {
        drop(x);  // Makes x unusable after the fact.
    });
    // println!("Value of x after return {:?}", x);
}
```

```rust,ignore
_6 = [closure@move.rs:7:13: 9:6] { x: move _1 }; // bb16[3]: scope 1 at move.rs:7:13: 9:6
```
Here, `x` is directly moved into the closure and the access to it will not be permitted after the
closure.

## Inferences in the compiler

Now let's dive into rustc code and see how all these inferences are done by the compiler.

Let's start with defining a term that we will be using quite a bit in the rest of the discussion -
*upvar*. An **upvar** is a variable that is local to the function where the closure is defined. So,
in the above examples, **x** will be an upvar to the closure. They are also sometimes referred to as
the *free variables* meaning they are not bound to the context of the closure.
[`compiler/rustc_passes/src/upvars.rs`][upvars] defines a query called *upvars_mentioned*
for this purpose.

[upvars]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_passes/upvars/index.html

Other than lazy invocation, one other thing that distinguishes a closure from a
normal function is that it can use the upvars. It borrows these upvars from its surrounding
context; therefore the compiler has to determine the upvar's borrow type. The compiler starts with
assigning an immutable borrow type and lowers the restriction (that is, changes it from
**immutable** to **mutable** to **move**) as needed, based on the usage. In the Example 1 above, the
closure only uses the variable for printing but does not modify it in any way and therefore, in the
`mir_dump`, we find the borrow type for the upvar `x` to be immutable.  In example 2, however, the
closure modifies `x` and increments it by some value.  Because of this mutation, the compiler, which
started off assigning `x` as an immutable reference type, has to adjust it as a mutable reference.
Likewise in the third example, the closure drops the vector and therefore this requires the variable
`x` to be moved into the closure. Depending on the borrow kind, the closure has to implement the
appropriate trait: `Fn` trait for immutable borrow, `FnMut` for mutable borrow,
and `FnOnce` for move semantics.

Most of the code related to the closure is in the
[`compiler/rustc_hir_typeck/src/upvar.rs`][upvar] file and the data structures are
declared in the file [`compiler/rustc_middle/src/ty/mod.rs`][ty].

[upvar]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/upvar/index.html
[ty]:https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/index.html

Before we go any further, let's discuss how we can examine the flow of control through the rustc
codebase. For closures specifically, set the `RUSTC_LOG` env variable as below and collect the
output in a file:

```console
> RUSTC_LOG=rustc_hir_typeck::upvar rustc +stage1 -Z dump-mir=all \
    <.rs file to compile> 2> <file where the output will be dumped>
```

This uses the stage1 compiler and enables `debug!` logging for the
`rustc_hir_typeck::upvar` module.

The other option is to step through the code using lldb or gdb.

1. `rust-lldb build/host/stage1/bin/rustc test.rs`
2. In lldb:
    1. `b upvar.rs:134`  // Setting the breakpoint on a certain line in the upvar.rs file
    2. `r`  // Run the program until it hits the breakpoint

Let's start with [`upvar.rs`][upvar]. This file has something called
the [`euv::ExprUseVisitor`] which walks the source of the closure and
invokes a callback for each upvar that is borrowed, mutated, or moved.

[`euv::ExprUseVisitor`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/expr_use_visitor/struct.ExprUseVisitor.html

```rust
fn main() {
    let mut x = vec![21];
    let _cl = || {
        let y = x[0];  // 1.
        x[0] += 1;  // 2.
    };
}
```

In the above example, our visitor will be called twice, for the lines marked 1 and 2, once for a
shared borrow and another one for a mutable borrow. It will also tell us what was borrowed.

The callbacks are defined by implementing the [`Delegate`] trait. The
[`InferBorrowKind`][ibk] type implements `Delegate` and keeps a map that
records for each upvar which mode of capture was required. The modes of capture
can be `ByValue` (moved) or `ByRef` (borrowed). For `ByRef` borrows, the possible
[`BorrowKind`]s are `ImmBorrow`, `UniqueImmBorrow`, `MutBorrow` as defined in the
[`compiler/rustc_middle/src/ty/mod.rs`][middle_ty].

[`BorrowKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.BorrowKind.html
[middle_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/index.html

`Delegate` defines a few different methods (the different callbacks):
**consume** for *move* of a variable, **borrow** for a *borrow* of some kind
(shared or mutable), and **mutate** when we see an *assignment* of something.

All of these callbacks have a common argument *cmt* which stands for Category,
Mutability and Type and is defined in
[`compiler/rustc_hir_typeck/src/expr_use_visitor.rs`][cmt]. Borrowing from the code
comments, "`cmt` is a complete categorization of a value indicating where it
originated and how it is located, as well as the mutability of the memory in
which the value is stored". Based on the callback (consume, borrow etc.), we
will call the relevant `adjust_upvar_borrow_kind_for_<something>` and pass the
`cmt` along. Once the borrow type is adjusted, we store it in the table, which
basically says what borrows were made for each closure.

```rust,ignore
self.tables
    .borrow_mut()
    .upvar_capture_map
    .extend(delegate.adjust_upvar_captures);
```

[`Delegate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/expr_use_visitor/trait.Delegate.html
[ibk]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/upvar/struct.InferBorrowKind.html
[cmt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/expr_use_visitor/index.html
