% The Stack and the Heap

As a systems language, Rust operates at a low level. If you’re coming from a
high-level language, there are some aspects of systems programming that you may
not be familiar with. The most important one is how memory works, with a stack
and a heap. If you’re familiar with how C-like languages use stack allocation,
this chapter will be a refresher. If you’re not, you’ll learn about this more
general concept, but with a Rust-y focus.

As with most things, when learning about them, we’ll use a simplified model to
start. This lets you get a handle on the basics, without getting bogged down
with details which are, for now, irrelevant. The examples we’ll use aren’t 100%
accurate, but are representative for the level we’re trying to learn at right
now. Once you have the basics down, learning more about how allocators are
implemented, virtual memory, and other advanced topics will reveal the leaks in
this particular abstraction.

# Memory management

These two terms are about memory management. The stack and the heap are
abstractions that help you determine when to allocate and deallocate memory.

Here’s a high-level comparison:

The stack is very fast, and is where memory is allocated in Rust by default.
But the allocation is local to a function call, and is limited in size. The
heap, on the other hand, is slower, and is explicitly allocated by your
program. But it’s effectively unlimited in size, and is globally accessible.

# The Stack

Let’s talk about this Rust program:

```rust
fn main() {
    let x = 42;
}
```

This program has one variable binding, `x`. This memory needs to be allocated
from somewhere. Rust ‘stack allocates’ by default, which means that basic
values ‘go on the stack’. What does that mean?

Well, when a function gets called, some memory gets allocated for all of its
local variables and some other information. This is called a ‘stack frame’, and
for the purpose of this tutorial, we’re going to ignore the extra information
and only consider the local variables we’re allocating. So in this case, when
`main()` is run, we’ll allocate a single 32-bit integer for our stack frame.
This is automatically handled for you, as you can see; we didn’t have to write
any special Rust code or anything.

When the function exits, its stack frame gets deallocated. This happens
automatically as well.

That’s all there is for this simple program. The key thing to understand here
is that stack allocation is very, very fast. Since we know all the local
variables we have ahead of time, we can grab the memory all at once. And since
we’ll throw them all away at the same time as well, we can get rid of it very
fast too.

The downside is that we can’t keep values around if we need them for longer
than a single function. We also haven’t talked about what the word, ‘stack’,
means. To do that, we need a slightly more complicated example:

```rust
fn foo() {
    let y = 5;
    let z = 100;
}

fn main() {
    let x = 42;

    foo();
}
```

This program has three variables total: two in `foo()`, one in `main()`. Just
as before, when `main()` is called, a single integer is allocated for its stack
frame. But before we can show what happens when `foo()` is called, we need to
visualize what’s going on with memory. Your operating system presents a view of
memory to your program that’s pretty simple: a huge list of addresses, from 0
to a large number, representing how much RAM your computer has. For example, if
you have a gigabyte of RAM, your addresses go from `0` to `1,073,741,823`. That
number comes from 2<sup>30</sup>, the number of bytes in a gigabyte. [^gigabyte]

[^gigabyte]: ‘Gigabyte’ can mean two things: 10^9, or 2^30. The SI standard resolved this by stating that ‘gigabyte’ is 10^9, and ‘gibibyte’ is 2^30. However, very few people use this terminology, and rely on context to differentiate. We follow in that tradition here.

This memory is kind of like a giant array: addresses start at zero and go
up to the final number. So here’s a diagram of our first stack frame:

| Address | Name | Value |
|---------|------|-------|
| 0       | x    | 42    |

We’ve got `x` located at address `0`, with the value `42`.

When `foo()` is called, a new stack frame is allocated:

| Address | Name | Value |
|---------|------|-------|
| 2       | z    | 100   |
| 1       | y    | 5     |
| 0       | x    | 42    |

Because `0` was taken by the first frame, `1` and `2` are used for `foo()`’s
stack frame. It grows upward, the more functions we call.


There are some important things we have to take note of here. The numbers 0, 1,
and 2 are all solely for illustrative purposes, and bear no relationship to the
address values the computer will use in reality. In particular, the series of
addresses are in reality going to be separated by some number of bytes that
separate each address, and that separation may even exceed the size of the
value being stored.

After `foo()` is over, its frame is deallocated:

| Address | Name | Value |
|---------|------|-------|
| 0       | x    | 42    |

And then, after `main()`, even this last value goes away. Easy!

It’s called a ‘stack’ because it works like a stack of dinner plates: the first
plate you put down is the last plate to pick back up. Stacks are sometimes
called ‘last in, first out queues’ for this reason, as the last value you put
on the stack is the first one you retrieve from it.

Let’s try a three-deep example:

```rust
fn italic() {
    let i = 6;
}

fn bold() {
    let a = 5;
    let b = 100;
    let c = 1;

    italic();
}

fn main() {
    let x = 42;

    bold();
}
```

We have some kooky function names to make the diagrams clearer.

Okay, first, we call `main()`:

| Address | Name | Value |
|---------|------|-------|
| 0       | x    | 42    |

Next up, `main()` calls `bold()`:

| Address | Name | Value |
|---------|------|-------|
| **3**   | **c**|**1**  |
| **2**   | **b**|**100**|
| **1**   | **a**| **5** |
| 0       | x    | 42    |

And then `bold()` calls `italic()`:

| Address | Name | Value |
|---------|------|-------|
| *4*     | *i*  | *6*   |
| **3**   | **c**|**1**  |
| **2**   | **b**|**100**|
| **1**   | **a**| **5** |
| 0       | x    | 42    |
Whew! Our stack is growing tall.

After `italic()` is over, its frame is deallocated, leaving only `bold()` and
`main()`:

| Address | Name | Value |
|---------|------|-------|
| **3**   | **c**|**1**  |
| **2**   | **b**|**100**|
| **1**   | **a**| **5** |
| 0       | x    | 42    | 

And then `bold()` ends, leaving only `main()`:

| Address | Name | Value |
|---------|------|-------|
| 0       | x    | 42    |

And then we’re done. Getting the hang of it? It’s like piling up dishes: you
add to the top, you take away from the top.

# The Heap

Now, this works pretty well, but not everything can work like this. Sometimes,
you need to pass some memory between different functions, or keep it alive for
longer than a single function’s execution. For this, we can use the heap.

In Rust, you can allocate memory on the heap with the [`Box<T>` type][box].
Here’s an example:

```rust
fn main() {
    let x = Box::new(5);
    let y = 42;
}
```

[box]: ../std/boxed/index.html

Here’s what happens in memory when `main()` is called:

| Address | Name | Value  |
|---------|------|--------|
| 1       | y    | 42     |
| 0       | x    | ?????? |

We allocate space for two variables on the stack. `y` is `42`, as it always has
been, but what about `x`? Well, `x` is a `Box<i32>`, and boxes allocate memory
on the heap. The actual value of the box is a structure which has a pointer to
‘the heap’. When we start executing the function, and `Box::new()` is called,
it allocates some memory for the heap, and puts `5` there. The memory now looks
like this:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 5                      |
| ...                  | ...  | ...                    |
| 1                    | y    | 42                     |
| 0                    | x    | → (2<sup>30</sup>) - 1 |

We have (2<sup>30</sup>) - 1 addresses in our hypothetical computer with 1GB of RAM. And since
our stack grows from zero, the easiest place to allocate memory is from the
other end. So our first value is at the highest place in memory. And the value
of the struct at `x` has a [raw pointer][rawpointer] to the place we’ve
allocated on the heap, so the value of `x` is (2<sup>30</sup>) - 1, the memory
location we’ve asked for.

[rawpointer]: raw-pointers.html

We haven’t really talked too much about what it actually means to allocate and
deallocate memory in these contexts. Getting into very deep detail is out of
the scope of this tutorial, but what’s important to point out here is that
the heap isn’t a stack that grows from the opposite end. We’ll have an
example of this later in the book, but because the heap can be allocated and
freed in any order, it can end up with ‘holes’. Here’s a diagram of the memory
layout of a program which has been running for a while now:


| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 5                      |
| (2<sup>30</sup>) - 2 |      |                        |
| (2<sup>30</sup>) - 3 |      |                        |
| (2<sup>30</sup>) - 4 |      | 42                     |
| ...                  | ...  | ...                    |
| 3                    | y    | → (2<sup>30</sup>) - 4 |
| 2                    | y    | 42                     |
| 1                    | y    | 42                     |
| 0                    | x    | → (2<sup>30</sup>) - 1 |

In this case, we’ve allocated four things on the heap, but deallocated two of
them. There’s a gap between (2<sup>30</sup>) - 1 and (2<sup>30</sup>) - 4 which isn’t
currently being used. The specific details of how and why this happens depends
on what kind of strategy you use to manage the heap. Different programs can use
different ‘memory allocators’, which are libraries that manage this for you.
Rust programs use [jemalloc][jemalloc] for this purpose.

[jemalloc]: http://www.canonware.com/jemalloc/

Anyway, back to our example. Since this memory is on the heap, it can stay
alive longer than the function which allocates the box. In this case, however,
it doesn’t.[^moving] When the function is over, we need to free the stack frame
for `main()`. `Box<T>`, though, has a trick up its sleeve: [Drop][drop]. The
implementation of `Drop` for `Box` deallocates the memory that was allocated
when it was created. Great! So when `x` goes away, it first frees the memory
allocated on the heap:

| Address | Name | Value  |
|---------|------|--------|
| 1       | y    | 42     |
| 0       | x    | ?????? |

[drop]: drop.html
[^moving]: We can make the memory live longer by transferring ownership,
           sometimes called ‘moving out of the box’. More complex examples will
           be covered later.


And then the stack frame goes away, freeing all of our memory.

# Arguments and borrowing

We’ve got some basic examples with the stack and the heap going, but what about
function arguments and borrowing? Here’s a small Rust program:

```rust
fn foo(i: &i32) {
    let z = 42;
}

fn main() {
    let x = 5;
    let y = &x;

    foo(y);
}
```

When we enter `main()`, memory looks like this:

| Address | Name | Value  |
|---------|------|--------|
| 1       | y    | → 0    |
| 0       | x    | 5      |

`x` is a plain old `5`, and `y` is a reference to `x`. So its value is the
memory location that `x` lives at, which in this case is `0`.

What about when we call `foo()`, passing `y` as an argument?

| Address | Name | Value  |
|---------|------|--------|
| 3       | z    | 42     |
| 2       | i    | → 0    |
| 1       | y    | → 0    |
| 0       | x    | 5      |

Stack frames aren’t only for local bindings, they’re for arguments too. So in
this case, we need to have both `i`, our argument, and `z`, our local variable
binding. `i` is a copy of the argument, `y`. Since `y`’s value is `0`, so is
`i`’s.

This is one reason why borrowing a variable doesn’t deallocate any memory: the
value of a reference is a pointer to a memory location. If we got rid of
the underlying memory, things wouldn’t work very well.

# A complex example

Okay, let’s go through this complex program step-by-step:

```rust
fn foo(x: &i32) {
    let y = 10;
    let z = &y;

    baz(z);
    bar(x, z);
}

fn bar(a: &i32, b: &i32) {
    let c = 5;
    let d = Box::new(5);
    let e = &d;

    baz(e);
}

fn baz(f: &i32) {
    let g = 100;
}

fn main() {
    let h = 3;
    let i = Box::new(20);
    let j = &h;

    foo(j);
}
```

First, we call `main()`:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| ...                  | ...  | ...                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

We allocate memory for `j`, `i`, and `h`. `i` is on the heap, and so has a
value pointing there.

Next, at the end of `main()`, `foo()` gets called:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| ...                  | ...  | ...                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

Space gets allocated for `x`, `y`, and `z`. The argument `x` has the same value
as `j`, since that’s what we passed it in. It’s a pointer to the `0` address,
since `j` points at `h`.

Next, `foo()` calls `baz()`, passing `z`:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| ...                  | ...  | ...                    |
| 7                    | g    | 100                    |
| 6                    | f    | → 4                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

We’ve allocated memory for `f` and `g`. `baz()` is very short, so when it’s
over, we get rid of its stack frame:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| ...                  | ...  | ...                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

Next, `foo()` calls `bar()` with `x` and `z`:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| (2<sup>30</sup>) - 2 |      | 5                      |
| ...                  | ...  | ...                    |
| 10                   | e    | → 9                    |
| 9                    | d    | → (2<sup>30</sup>) - 2 |
| 8                    | c    | 5                      |
| 7                    | b    | → 4                    |
| 6                    | a    | → 0                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

We end up allocating another value on the heap, and so we have to subtract one
from (2<sup>30</sup>) - 1. It’s easier to write that than `1,073,741,822`. In any
case, we set up the variables as usual.

At the end of `bar()`, it calls `baz()`:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| (2<sup>30</sup>) - 2 |      | 5                      |
| ...                  | ...  | ...                    |
| 12                   | g    | 100                    |
| 11                   | f    | → (2<sup>30</sup>) - 2 |
| 10                   | e    | → 9                    |
| 9                    | d    | → (2<sup>30</sup>) - 2 |
| 8                    | c    | 5                      |
| 7                    | b    | → 4                    |
| 6                    | a    | → 0                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

With this, we’re at our deepest point! Whew! Congrats for following along this
far.

After `baz()` is over, we get rid of `f` and `g`:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| (2<sup>30</sup>) - 2 |      | 5                      |
| ...                  | ...  | ...                    |
| 10                   | e    | → 9                    |
| 9                    | d    | → (2<sup>30</sup>) - 2 |
| 8                    | c    | 5                      |
| 7                    | b    | → 4                    |
| 6                    | a    | → 0                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

Next, we return from `bar()`. `d` in this case is a `Box<T>`, so it also frees
what it points to: (2<sup>30</sup>) - 2.

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| ...                  | ...  | ...                    |
| 5                    | z    | → 4                    |
| 4                    | y    | 10                     |
| 3                    | x    | → 0                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

And after that, `foo()` returns:

| Address              | Name | Value                  |
|----------------------|------|------------------------|
| (2<sup>30</sup>) - 1 |      | 20                     |
| ...                  | ...  | ...                    |
| 2                    | j    | → 0                    |
| 1                    | i    | → (2<sup>30</sup>) - 1 |
| 0                    | h    | 3                      |

And then, finally, `main()`, which cleans the rest up. When `i` is `Drop`ped,
it will clean up the last of the heap too.

# What do other languages do?

Most languages with a garbage collector heap-allocate by default. This means
that every value is boxed. There are a number of reasons why this is done, but
they’re out of scope for this tutorial. There are some possible optimizations
that don’t make it true 100% of the time, too. Rather than relying on the stack
and `Drop` to clean up memory, the garbage collector deals with the heap
instead.

# Which to use?

So if the stack is faster and easier to manage, why do we need the heap? A big
reason is that Stack-allocation alone means you only have 'Last In First Out (LIFO)' semantics for
reclaiming storage. Heap-allocation is strictly more general, allowing storage
to be taken from and returned to the pool in arbitrary order, but at a
complexity cost.

Generally, you should prefer stack allocation, and so, Rust stack-allocates by
default. The LIFO model of the stack is simpler, at a fundamental level. This
has two big impacts: runtime efficiency and semantic impact.

## Runtime Efficiency

Managing the memory for the stack is trivial: The machine
increments or decrements a single value, the so-called “stack pointer”.
Managing memory for the heap is non-trivial: heap-allocated memory is freed at
arbitrary points, and each block of heap-allocated memory can be of arbitrary
size, the memory manager must generally work much harder to identify memory for
reuse.

If you’d like to dive into this topic in greater detail, [this paper][wilson]
is a great introduction.

[wilson]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.143.4688

## Semantic impact

Stack-allocation impacts the Rust language itself, and thus the developer’s
mental model. The LIFO semantics is what drives how the Rust language handles
automatic memory management. Even the deallocation of a uniquely-owned
heap-allocated box can be driven by the stack-based LIFO semantics, as
discussed throughout this chapter. The flexibility (i.e. expressiveness) of non
LIFO-semantics means that in general the compiler cannot automatically infer at
compile-time where memory should be freed; it has to rely on dynamic protocols,
potentially from outside the language itself, to drive deallocation (reference
counting, as used by `Rc<T>` and `Arc<T>`, is one example of this).

When taken to the extreme, the increased expressive power of heap allocation
comes at the cost of either significant runtime support (e.g. in the form of a
garbage collector) or significant programmer effort (in the form of explicit
memory management calls that require verification not provided by the Rust
compiler).

