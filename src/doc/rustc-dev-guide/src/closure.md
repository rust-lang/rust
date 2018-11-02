# Closure Expansion in rustc

Let's start with a few examples

### Example 1
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
Let's say the above is the content of a file called immut.rs. If we compile immut.rs using the command
```
rustc +stage1 immut.rs -Zdump-mir=all
```
we will see a newly generated directory in our current working directory called mir_dump, which will
contain several files. If we look at file `rustc.main.-------.mir_map.0.mir`, we will find, among
other things, it also contains this line:

```rust,ignore
_4 = &_1;                        // bb0[6]: scope 1 at immut.rs:7:13: 7:36
_3 = [closure@immut.rs:7:13: 7:36] { x: move _4 }; // bb0[7]: scope 1 at immut.rs:7:13: 7:36
```
Here in first line `_4 = &_1;`, the mir_dump tells us that x was borrowed as an immutable reference.
This is what we would hope as our closure just reads x.

### Example 2
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
_4 = &mut _1;                    // bb0[6]: scope 1 at mut.rs:7:13: 10:6
_3 = [closure@mut.rs:7:13: 10:6] { x: move _4 }; // bb0[7]: scope 1 at mut.rs:7:13: 10:6
```
This time along, in the line `_4 = &mut _1;`, we see that the borrow is changed to mutable borrow.
fair enough as the closure increments x by 10.

### Example 3
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
Here, x is directly moved into the closure and the access to it will not be permitted after the
closure.


Now let's dive into rustc code and see how all these inferences are done by the compiler.

Let's start with defining a term that we will be using quite a bit in the rest of the discussion -
*upvar*. An **upvar** is a variable that is local to the function, where the closure is defined. So,
in the above examples, **x** will be an upvar to the closure. They are also sometimes referred to as
the *free variables* meaning they are not bound to the context of the closure.
`src/librustc/ty/query/mod.rs` defines a query called *freevars* for this purpose.

So, we know that other than lazy invocation, one other thing that the distinguishes a closure from a
normal function is that it can use the upvars. Because, it borrows these upvars from its surrounding
context, therfore the compiler has to determine the upvar's borrow type. The compiler starts with
assigning an immutable borrow type and lowers the restriction (that is, changes it from
**immutable** to **mutable** to **move**) as needed, based on the usage. In the Example 1 above, the
closure only uses the variable for printing but does not modify it in any way and therefore, in the
mir_dump, we find the borrow type for the upvar x to be immutable.  In example 2, however the
closure modifies x and increments it by some value.  Because of this mutation, the compiler, which
started off assigning x as an immutable reference type, has to adjust it as mutable reference.
Likewise in the third example, the closure drops the vector and therefore this requires the variable
x to be moved into the closure. Depending on the borrow kind, the closure has to implement the
appropriate trait.  Fn trait for immutable borrow, FnMut for mutable borrow and FnOnce for move
semantics.

Most of the code related to the closure is in the src/librustc_typeck/check/upvar.rs file and the
data structures are declared in the file src/librustc/ty/mod.rs.

Before we go any further, let's discuss how we can examine the flow of coontrol through the rustc
codebase. For the closure part specifically, I would set the RUST_LOG as under and collect the
output in a file

```
RUST_LOG=rustc_typeck::check::upvar rustc +stage1 -Zdump-mir=all <.rs file to compile> 2> <file
where the output will be dumped>
```

This uses the stage1 compiler.

The other option is to step through the code using lldb or gdb.

```
1. rust-lldb build/x86_64-apple-darwin/stage1/bin/rustc test.rs
2. b upvar.rs:134  // Setting the breakpoint on a certain line in the upvar.rs file
3. r  // Run the program until it hits the breakpoint
```

Let's start with the file: `upvar.rs`.  This file has something called the euv::ExprUseVisitor which
walks the source of the closure and it gets called back for each upvar that is borrowed, mutated or
moved.

```rust
fn main() {
    let x = vec![21];
    let _cl = || {
        let y = x[0];  // 1.
        x[0] += 1;  // 2.
    };
}
```

In the above example, our visitor will be called twice, for the lines marked 1 and 2, once as a
shared borrow and another one as a mutable borrow. It will also tell as what was borrowed. The
callbacks get invoked at the delegate. The delegate is of type `struct InferBorrowKind` which has a
few fields but the one we are interested in is the `adjust_upvar_captures` which is of type
`FxHashMap<UpvarId, UpvarCapture<'tcx>>` which tells us for each upvar, which mode of borrow did we
require. The modes of borrow can be ByValue (moved) or ByRef (borrowed) and for ByRef borrows, it
can be one among shared, shallow, unique or mut as defined in the `src/librustc/mir/mod.rs`

The method callbacks are the method implementations of the euv::Delegate trait for InferBorrowKind.
**consume** callback is for *move* of a variable, **borrow** callback if there is a *borrow* of some
kind, shared or mutable and **mutate** when we see an *assignment* of something. We will see that
all these callbacks have a common argument *cmt* which stands for category, Mutability and Type and
is defined in *src/librustc/middle/mem_categorization.rs*. Borrowing from the code comments *cmt *is
a complete categorization of a value indicating where it originated and how it is located, as well
as the mutability of the memory in which the value is stored.** Based on the callback (consume,
borrow etc.), we will call the relevant *adjust_upvar_borrow_kind_for_<something>* and pass the cmt
along. Once the borrow type is adjusted, we store it in the table, which basically says for this
closure, these set of borrows were made.

```
self.tables
    .borrow_mut()
    .upvar_capture_map
    .extend(delegate.adjust_upvar_captures);
```
