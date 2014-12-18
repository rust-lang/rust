% The Rust Ownership Guide

This guide presents Rust's ownership system. This is one of Rust's most unique
and compelling features, with which Rust developers should become quite
acquainted. Ownership is how Rust achieves its largest goal, memory safety.
The ownership system has a few distinct concepts: **ownership**, **borrowing**,
and **lifetimes**. We'll talk about each one in turn.

# Meta

Before we get to the details, two important notes about the ownership system.

Rust has a focus on safety and speed. It accomplishes these goals through many
"zero cost abstractions," which means that in Rust, abstractions cost as little
as possible in order to make them work. The ownership system is a prime example
of a zero cost abstraction. All of the analysis we'll talk about in this guide
is _done at compile time_. You do not pay any run-time cost for any of these
features.

However, this system does have a certain cost: learning curve. Many new users
to Rust experience something we like to call "fighting with the borrow
checker," where the Rust compiler refuses to compile a program that the author
thinks is valid. This often happens because the programmer's mental model of
how ownership should work doesn't match the actual rules that Rust implements.
You probably will experience similar things at first. There is good news,
however: more experienced Rust developers report that once they work with the
rules of the ownership system for a period of time, they fight the borrow
checker less and less.

With that in mind, let's learn about ownership.

# Ownership

At its core, ownership is about 'resources.' For the purposes of the vast
majority of this guide, we will talk about a specific resource: memory. The
concept generalizes to any kind of resource, like a file handle, but to make it
more concrete, we'll focus on memory.

When your program allocates some memory, it needs some way to deallocate that
memory. Imagine a function `foo` that allocates four bytes of memory, and then
never deallocates that memory. We call this problem 'leaking' memory, because
each time we call `foo`, we're allocating another four bytes. Eventually, with
enough calls to `foo`, we will run our system out of memory. That's no good. So
we need some way for `foo` to deallocate those four bytes. It's also important
that we don't deallocate too many times, either. Without getting into the
details, attempting to deallocate memory multiple times can lead to problems.
In other words, any time some memory is allocated, we need to make sure that we
deallocate that memory once and only once. Too many times is bad, not enough
times is bad. The counts must match.

There's one other important detail with regards to allocating memory. Whenever
we request some amount of memory, what we are given is a handle to that memory.
This handle (often called a 'pointer', when we're referring to memory) is how
we interact with the allocated memory. As long as we have that handle, we can
do something with the memory. Once we're done with the handle, we're also done
with the memory, as we can't do anything useful without a handle to it.

Historically, systems programming languages require you to track these
allocations, deallocations, and handles yourself. For example, if we want some
memory from the heap in a language like C, we do this:

```c
{
    int *x = malloc(sizeof(int));

    // we can now do stuff with our handle x
    *x = 5;

    free(x);
}
```

The call to `malloc` allocates some memory. The call to `free` deallocates the
memory. There's also bookkeeping about allocating the correct amount of memory.

Rust combines these two aspects of allocating memory (and other resources) into
a concept called 'ownership.' Whenever we request some memory, that handle we
receive is called the 'owning handle.' Whenever that handle goes out of scope,
Rust knows that you cannot do anything with the memory anymore, and so
therefore deallocates the memory for you. Here's the equivalent example in
Rust:

```rust
{
    let x = box 5i;
}
```

The `box` keyword creates a `Box<T>` (specifically `Box<int>` in this case) by
allocating a small segment of memory on the heap with enough space to fit an
`int`. But where in the code is the box deallocated? We said before that we
must have a deallocation for each allocation. Rust handles this for you. It
knows that our handle, `x`, is the owning reference to our box. Rust knows that
`x` will go out of scope at the end of the block, and so it inserts a call to
deallocate the memory at the end of the scope. Because the compiler does this
for us, it's impossible to forget. We always have exactly one deallocation paired
with each of our allocations.

This is pretty straightforward, but what happens when we want to pass our box
to a function? Let's look at some code:

```rust
fn main() {
    let x = box 5i;

    add_one(x);
}

fn add_one(mut num: Box<int>) {
    *num += 1;
}
```

This code works, but it's not ideal. For example, let's add one more line of
code, where we print out the value of `x`:

```{rust,ignore}
fn main() {
    let x = box 5i;

    add_one(x);

    println!("{}", x);
}

fn add_one(mut num: Box<int>) {
    *num += 1;
}
```

This does not compile, and gives us an error:

```text
error: use of moved value: `x`
   println!("{}", x);
                  ^
```

Remember, we need one deallocation for every allocation. When we try to pass
our box to `add_one`, we would have two handles to the memory: `x` in `main`,
and `num` in `add_one`. If we deallocated the memory when each handle went out
of scope, we would have two deallocations and one allocation, and that's wrong.
So when we call `add_one`, Rust defines `num` as the owner of the handle. And
so, now that we've given ownership to `num`, `x` is invalid. `x`'s value has
"moved" from `x` to `num`. Hence the error: use of moved value `x`.

To fix this, we can have `add_one` give ownership back when it's done with the
box:

```rust
fn main() {
    let x = box 5i;

    let y = add_one(x);

    println!("{}", y);
}

fn add_one(mut num: Box<int>) -> Box<int> {
    *num += 1;

    num
}
```

This code will compile and run just fine. Now, we return a `box`, and so the
ownership is transferred back to `y` in `main`. We only have ownership for the
duration of our function before giving it back. This pattern is very common,
and so Rust introduces a concept to describe a handle which temporarily refers
to something another handle owns. It's called "borrowing," and it's done with
"references", designated by the `&` symbol.

# Borrowing

Here's the current state of our `add_one` function:

```rust
fn add_one(mut num: Box<int>) -> Box<int> {
    *num += 1;

    num
}
```

This function takes ownership, because it takes a `Box`, which owns its
contents. But then we give ownership right back.

In the physical world, you can give one of your possessions to someone for a
short period of time. You still own your possession, you're just letting someone
else use it for a while. We call that 'lending' something to someone, and that
person is said to be 'borrowing' that something from you.

Rust's ownership system also allows an owner to lend out a handle for a limited
period. This is also called 'borrowing.' Here's a version of `add_one` which
borrows its argument rather than taking ownership:

```rust
fn add_one(num: &mut int) {
    *num += 1;
}
```

This function borrows an `int` from its caller, and then increments it. When
the function is over, and `num` goes out of scope, the borrow is over.

# Lifetimes

Lending out a reference to a resource that someone else owns can be
complicated, however. For example, imagine this set of operations:

1. I acquire a handle to some kind of resource.
2. I lend you a reference to the resource.
3. I decide I'm done with the resource, and deallocate it, while you still have
   your reference.
4. You decide to use the resource.

Uh oh! Your reference is pointing to an invalid resource. This is called a
"dangling pointer" or "use after free," when the resource is memory.

To fix this, we have to make sure that step four never happens after step
three. The ownership system in Rust does this through a concept called
"lifetimes," which describe the scope that a reference is valid for.

Let's look at that function which borrows an `int` again:

```rust
fn add_one(num: &int) -> int {
    *num + 1
}
```

Rust has a feature called 'lifetime elision,' which allows you to not write
lifetime annotations in certain circumstances. This is one of them. Without
eliding the lifetimes, `add_one` looks like this:

```rust
fn add_one<'a>(num: &'a int) -> int {
    *num + 1
}
```

The `'a` is called a **lifetime**. Most lifetimes are used in places where
short names like `'a`, `'b` and `'c` are clearest, but it's often useful to
have more descriptive names. Let's dig into the syntax in a bit more detail:

```{rust,ignore}
fn add_one<'a>(...)
```

This part _declares_ our lifetimes. This says that `add_one` has one lifetime,
`'a`. If we had two, it would look like this:

```{rust,ignore}
fn add_two<'a, 'b>(...)
```

Then in our parameter list, we use the lifetimes we've named:

```{rust,ignore}
...(num: &'a int) -> ...
```

If you compare `&int` to `&'a int`, they're the same, it's just that the
lifetime `'a` has snuck in between the `&` and the `int`. We read `&int` as "a
reference to an int" and `&'a int` as "a reference to an int with the lifetime 'a.'"

Why do lifetimes matter? Well, for example, here's some code:

```rust
struct Foo<'a> {
    x: &'a int,
}

fn main() {
    let y = &5i; // this is the same as `let _y = 5; let y = &_y;
    let f = Foo { x: y };

    println!("{}", f.x);
}
```

As you can see, `struct`s can also have lifetimes. In a similar way to functions,

```{rust}
struct Foo<'a> {
# x: &'a int,
# }
```

declares a lifetime, and

```rust
# struct Foo<'a> {
x: &'a int,
# }
```

uses it. So why do we need a lifetime here? We need to ensure that any reference
to a `Foo` cannot outlive the reference to an `int` it contains.

## Thinking in scopes

A way to think about lifetimes is to visualize the scope that a reference is
valid for. For example:

```rust
fn main() {
    let y = &5i;    // -+ y goes into scope
                    //  |
    // stuff        //  |
                    //  |
}                   // -+ y goes out of scope
```

Adding in our `Foo`:

```rust
struct Foo<'a> {
    x: &'a int,
}

fn main() {
    let y = &5i;          // -+ y goes into scope
    let f = Foo { x: y }; // -+ f goes into scope
    // stuff              //  |
                          //  |
}                         // -+ f and y go out of scope
```

Our `f` lives within the scope of `y`, so everything works. What if it didn't?
This code won't work:

```{rust,ignore}
struct Foo<'a> {
    x: &'a int,
}

fn main() {
    let x;                    // -+ x goes into scope
                              //  |
    {                         //  |
        let y = &5i;          // ---+ y goes into scope
        let f = Foo { x: y }; // ---+ f goes into scope
        x = &f.x;             //  | | error here
    }                         // ---+ f and y go out of scope
                              //  |
    println!("{}", x);        //  |
}                             // -+ x goes out of scope
```

Whew! As you can see here, the scopes of `f` and `y` are smaller than the scope
of `x`. But when we do `x = &f.x`, we make `x` a reference to something that's
about to go out of scope.

Named lifetimes are a way of giving these scopes a name. Giving something a
name is the first step towards being able to talk about it.

## 'static

The lifetime named 'static' is a special lifetime. It signals that something
has the lifetime of the entire program. Most Rust programmers first come across
`'static` when dealing with strings:

```rust
let x: &'static str = "Hello, world.";
```

String literals have the type `&'static str` because the reference is always
alive: they are baked into the data segment of the final binary. Another
example are globals:

```rust
static FOO: int = 5i;
let x: &'static int = &FOO;
```

This adds an `int` to the data segment of the binary, and FOO is a reference to
it.

# Shared Ownership

In all the examples we've considered so far, we've assumed that each handle has
a singular owner. But sometimes, this doesn't work. Consider a car. Cars have
four wheels. We would want a wheel to know which car it was attached to. But
this won't work:

```{rust,ignore}
struct Car {
    name: String,
}

struct Wheel {
    size: int,
    owner: Car,
}

fn main() {
    let car = Car { name: "DeLorean".to_string() };

    for _ in range(0u, 4) {
        Wheel { size: 360, owner: car };
    }
}
```

We try to make four `Wheel`s, each with a `Car` that it's attached to. But the
compiler knows that on the second iteration of the loop, there's a problem:

```text
error: use of moved value: `car`
    Wheel { size: 360, owner: car };
                              ^~~
note: `car` moved here because it has type `Car`, which is non-copyable
    Wheel { size: 360, owner: car };
                              ^~~
```

We need our `Car` to be pointed to by multiple `Wheel`s. We can't do that with
`Box<T>`, because it has a single owner. We can do it with `Rc<T>` instead:

```rust
use std::rc::Rc;

struct Car {
    name: String,
}

struct Wheel {
    size: int,
    owner: Rc<Car>,
}

fn main() {
    let car = Car { name: "DeLorean".to_string() };

    let car_owner = Rc::new(car);

    for _ in range(0u, 4) {
        Wheel { size: 360, owner: car_owner.clone() };
    }
}
```

We wrap our `Car` in an `Rc<T>`, getting an `Rc<Car>`, and then use the
`clone()` method to make new references. We've also changed our `Wheel` to have
an `Rc<Car>` rather than just a `Car`.

This is the simplest kind of multiple ownership possible. For example, there's
also `Arc<T>`, which uses more expensive atomic instructions to be the
thread-safe counterpart of `Rc<T>`.

# Related Resources

Coming Soon.
