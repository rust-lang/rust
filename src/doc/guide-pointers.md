% The Rust Pointer Guide

Rust's pointers are one of its more unique and compelling features. Pointers
are also one of the more confusing topics for newcomers to Rust. They can also
be confusing for people coming from other languages that support pointers, such
as C++. This guide will help you understand this important topic.

Be sceptical of non-reference pointers in Rust: use them for a deliberate
purpose, not just to make the compiler happy. Each pointer type comes with an
explanation about when they are appropriate to use. Default to references
unless you're in one of those specific situations.

You may be interested in the [cheat sheet](#cheat-sheet), which gives a quick
overview of the types, names, and purpose of the various pointers.

# An introduction

If you aren't familiar with the concept of pointers, here's a short
introduction.  Pointers are a very fundamental concept in systems programming
languages, so it's important to understand them.

## Pointer Basics

When you create a new variable binding, you're giving a name to a value that's
stored at a particular location on the stack. (If you're not familiar with the
"heap" vs. "stack", please check out [this Stack Overflow
question](http://stackoverflow.com/questions/79923/what-and-where-are-the-stack-and-heap),
as the rest of this guide assumes you know the difference.) Like this:

```{rust}
let x = 5i;
let y = 8i;
```
| location | value |
|----------|-------|
| 0xd3e030 | 5	   |
| 0xd3e028 | 8     |

We're making up memory locations here, they're just sample values. Anyway, the
point is that `x`, the name we're using for our variable, corresponds to the
memory location `0xd3e030`, and the value at that location is `5`. When we
refer to `x`, we get the corresponding value. Hence, `x` is `5`.

Let's introduce a pointer. In some languages, there is just one type of
'pointer,' but in Rust, we have many types. In this case, we'll use a Rust
**reference**, which is the simplest kind of pointer.

```{rust}
let x = 5i;
let y = 8i;
let z = &y;
```
|location | value    |
|-------- |----------|
|0xd3e030 | 5        |
|0xd3e028 | 8        |
|0xd3e020 | 0xd3e028 |

See the difference? Rather than contain a value, the value of a pointer is a
location in memory. In this case, the location of `y`. `x` and `y` have the
type `int`, but `z` has the type `&int`. We can print this location using the
`{:p}` format string:

```{rust}
let x = 5i;
let y = 8i;
let z = &y;

println!("{:p}", z);
```

This would print `0xd3e028`, with our fictional memory addresses.

Because `int` and `&int` are different types, we can't, for example, add them
together:

```{rust,ignore}
let x = 5i;
let y = 8i;
let z = &y;

println!("{}", x + z);
```

This gives us an error:

```{notrust,ignore}
hello.rs:6:24: 6:25 error: mismatched types: expected `int` but found `&int` (expected int but found &-ptr)
hello.rs:6     println!("{}", x + z);
                                  ^
```

We can **dereference** the pointer by using the `*` operator. Dereferencing a
pointer means accessing the value at the location stored in the pointer. This
will work:

```{rust}
let x = 5i;
let y = 8i;
let z = &y;

println!("{}", x + *z);
```

It prints `13`.

That's it! That's all pointers are: they point to some memory location. Not
much else to them. Now that we've discussed the 'what' of pointers, let's
talk about the 'why.'

## Pointer uses

Rust's pointers are quite useful, but in different ways than in other systems
languages. We'll talk about best practices for Rust pointers later in
the guide, but here are some ways that pointers are useful in other languages:

In C, strings are a pointer to a list of `char`s, ending with a null byte.
The only way to use strings is to get quite familiar with pointers.

Pointers are useful to point to memory locations that are not on the stack. For
example, our example used two stack variables, so we were able to give them
names. But if we allocated some heap memory, we wouldn't have that name
available.  In C, `malloc` is used to allocate heap memory, and it returns a
pointer.

As a more general variant of the previous two points, any time you have a
structure that can change in size, you need a pointer. You can't tell at
compile time how much memory to allocate, so you've gotta use a pointer to
point at the memory where it will be allocated, and deal with it at run time.

Pointers are useful in languages that are pass-by-value, rather than
pass-by-reference. Basically, languages can make two choices (this is made
up syntax, it's not Rust):

```{notrust,ignore}
fn foo(x) {
    x = 5
}

fn main() {
    i = 1
    foo(i)
    // what is the value of i here?
}
```

In languages that are pass-by-value, `foo` will get a copy of `i`, and so
the original version of `i` is not modified. At the comment, `i` will still be
`1`. In a language that is pass-by-reference, `foo` will get a reference to `i`,
and therefore, can change its value. At the comment, `i` will be `5`.

So what do pointers have to do with this? Well, since pointers point to a
location in memory...

```{notrust,ignore}
fn foo(&int x) {
    *x = 5
}

fn main() {
    i = 1
    foo(&i)
    // what is the value of i here?
}
```

Even in a language which is pass by value, `i` will be `5` at the comment. You
see, because the argument `x` is a pointer, we do send a copy over to `foo`,
but because it points at a memory location, which we then assign to, the
original value is still changed. This pattern is called
'pass-reference-by-value.' Tricky!

## Common pointer problems

We've talked about pointers, and we've sung their praises. So what's the
downside? Well, Rust attempts to mitigate each of these kinds of problems,
but here are problems with pointers in other languages:

Uninitialized pointers can cause a problem. For example, what does this program
do?

```{notrust,ignore}
&int x;
*x = 5; // whoops!
```

Who knows? We just declare a pointer, but don't point it at anything, and then
set the memory location that it points at to be `5`. But which location? Nobody
knows. This might be harmless, and it might be catastrophic.

When you combine pointers and functions, it's easy to accidentally invalidate
the memory the pointer is pointing to. For example:

```{notrust,ignore}
fn make_pointer(): &int {
    x = 5;

    return &x;
}

fn main() {
    &int i = make_pointer();
    *i = 5; // uh oh!
}
```

`x` is local to the `make_pointer` function, and therefore, is invalid as soon
as `make_pointer` returns. But we return a pointer to its memory location, and
so back in `main`, we try to use that pointer, and it's a very similar
situation to our first one. Setting invalid memory locations is bad.

As one last example of a big problem with pointers, **aliasing** can be an
issue. Two pointers are said to alias when they point at the same location
in memory. Like this:

```{notrust,ignore}
fn mutate(&int i, int j) {
    *i = j;
}

fn main() {
  x = 5;
  y = &x;
  z = &x; //y and z are aliased


  run_in_new_thread(mutate, y, 1);
  run_in_new_thread(mutate, z, 100);

  // what is the value of x here?
}
```

In this made-up example, `run_in_new_thread` spins up a new thread, and calls
the given function name with its arguments. Since we have two threads, and
they're both operating on aliases to `x`, we can't tell which one finishes
first, and therefore, the value of `x` is actually non-deterministic. Worse,
what if one of them had invalidated the memory location they pointed to? We'd
have the same problem as before, where we'd be setting an invalid location.

## Conclusion

That's a basic overview of pointers as a general concept. As we alluded to
before, Rust has different kinds of pointers, rather than just one, and
mitigates all of the problems that we talked about, too. This does mean that
Rust pointers are slightly more complicated than in other languages, but
it's worth it to not have the problems that simple pointers have.

# References

The most basic type of pointer that Rust has is called a 'reference.' Rust
references look like this:

```{rust}
let x = 5i;
let y = &x;

println!("{}", *y);
println!("{:p}", y);
println!("{}", y);
```

We'd say "`y` is a reference to `x`." The first `println!` prints out the
value of `y`'s referent by using the dereference operator, `*`. The second
one prints out the memory location that `y` points to, by using the pointer
format string. The third `println!` *also* prints out the value of `y`'s
referent, because `println!` will automatically dereference it for us.

Here's a function that takes a reference:

```{rust}
fn succ(x: &int) -> int { *x + 1 }
```

You can also use `&` as an operator to create a reference, so we can
call this function in two different ways:

```{rust}
fn succ(x: &int) -> int { *x + 1 }

fn main() {

    let x = 5i;
    let y = &x;

    println!("{}", succ(y));
    println!("{}", succ(&x));
}
```

Both of these `println!`s will print out `6`.

Of course, if this were real code, we wouldn't bother with the reference, and
just write:

```{rust}
fn succ(x: int) -> int { x + 1 }
```

References are immutable by default:

```{rust,ignore}
let x = 5i;
let y = &x;

*y = 5; // error: cannot assign to immutable dereference of `&`-pointer `*y`
```

They can be made mutable with `mut`, but only if its referent is also mutable.
This works:

```{rust}
let mut x = 5i;
let y = &mut x;
```

This does not:

```{rust,ignore}
let x = 5i;
let y = &mut x; // error: cannot borrow immutable local variable `x` as mutable
```

Immutable pointers are allowed to alias:

```{rust}
let x = 5i;
let y = &x;
let z = &x;
```

Mutable ones, however, are not:

```{rust,ignore}
let x = 5i;
let y = &mut x;
let z = &mut x; // error: cannot borrow `x` as mutable more than once at a time
```

Despite their complete safety, a reference's representation at runtime is the
same as that of an ordinary pointer in a C program. They introduce zero
overhead. The compiler does all safety checks at compile time. The theory that
allows for this was originally called **region pointers**. Region pointers
evolved into what we know today as **lifetimes**.

Here's the simple explanation: would you expect this code to compile?

```{rust,ignore}
fn main() {
    println!("{}", x);
    let x = 5;
}
```

Probably not. That's because you know that the name `x` is valid from where
it's declared to when it goes out of scope. In this case, that's the end of
the `main` function. So you know this code will cause an error. We call this
duration a 'lifetime'. Let's try a more complex example:

```{rust}
fn main() {
    let x = &mut 5i;

    if *x < 10 {
        let y = &x;

        println!("Oh no: {}", y);
        return;
    }

    *x -= 1;

    println!("Oh no: {}", x);
}
```

Here, we're borrowing a pointer to `x` inside of the `if`. The compiler, however,
is able to determine that that pointer will go out of scope without `x` being
mutated, and therefore, lets us pass. This wouldn't work:

```{rust,ignore}
fn main() {
    let x = &mut 5i;

    if *x < 10 {
        let y = &x;
        *x -= 1;

        println!("Oh no: {}", y);
        return;
    }

    *x -= 1;

    println!("Oh no: {}", x);
}
```

It gives this error:

```{notrust,ignore}
test.rs:5:8: 5:10 error: cannot assign to `*x` because it is borrowed
test.rs:5         *x -= 1;
                  ^~
test.rs:4:16: 4:18 note: borrow of `*x` occurs here
test.rs:4         let y = &x;
                          ^~
```

As you might guess, this kind of analysis is complex for a human, and therefore
hard for a computer, too! There is an entire [guide devoted to references
and lifetimes](guide-lifetimes.html) that goes into lifetimes in
great detail, so if you want the full details, check that out.

## Best practices

In general, prefer stack allocation over heap allocation. Using references to
stack allocated information is preferred whenever possible. Therefore,
references are the default pointer type you should use, unless you have
specific reason to use a different type. The other types of pointers cover when
they're appropriate to use in their own best practices sections.

Use references when you want to use a pointer, but do not want to take ownership.
References just borrow ownership, which is more polite if you don't need the
ownership. In other words, prefer:

```{rust}
fn succ(x: &int) -> int { *x + 1 }
```

to

```{rust}
fn succ(x: Box<int>) -> int { *x + 1 }
```

As a corollary to that rule, references allow you to accept a wide variety of
other pointers, and so are useful so that you don't have to write a number
of variants per pointer. In other words, prefer:

```{rust}
fn succ(x: &int) -> int { *x + 1 }
```

to

```{rust}
fn box_succ(x: Box<int>) -> int { *x + 1 }

fn rc_succ(x: std::rc::Rc<int>) -> int { *x + 1 }
```

# Boxes

`Box<T>` is Rust's 'boxed pointer' type. Boxes provide the simplest form of
heap allocation in Rust. Creating a box looks like this:

```{rust}
let x = box(std::boxed::HEAP) 5i;
```

`box` is a keyword that does 'placement new,' which we'll talk about in a bit.
`box` will be useful for creating a number of heap-allocated types, but is not
quite finished yet. In the meantime, `box`'s type defaults to
`std::boxed::HEAP`, and so you can leave it off:

```{rust}
let x = box 5i;
```

As you might assume from the `HEAP`, boxes are heap allocated. They are
deallocated automatically by Rust when they go out of scope:

```{rust}
{
    let x = box 5i;

    // stuff happens

} // x is destructed and its memory is free'd here
```

However, boxes do _not_ use reference counting or garbage collection. Boxes are
what's called an **affine type**. This means that the Rust compiler, at compile
time, determines when the box comes into and goes out of scope, and inserts the
appropriate calls there. Furthermore, boxes are a specific kind of affine type,
known as a **region**. You can read more about regions [in this paper on the
Cyclone programming
language](http://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf).

You don't need to fully grok the theory of affine types or regions to grok
boxes, though. As a rough approximation, you can treat this Rust code:

```{rust}
{
    let x = box 5i;

    // stuff happens
}
```

As being similar to this C code:

```{notrust,ignore}
{
    int *x;
    x = (int *)malloc(sizeof(int));

    // stuff happens

    free(x);
}
```

Of course, this is a 10,000 foot view. It leaves out destructors, for example.
But the general idea is correct: you get the semantics of `malloc`/`free`, but
with some improvements:

1. It's impossible to allocate the incorrect amount of memory, because Rust
   figures it out from the types.
2. You cannot forget to `free` memory you've allocated, because Rust does it
   for you.
3. Rust ensures that this `free` happens at the right time, when it is truly
   not used. Use-after-free is not possible.
4. Rust enforces that no other writeable pointers alias to this heap memory,
   which means writing to an invalid pointer is not possible.

See the section on references or the [lifetimes guide](guide-lifetimes.html)
for more detail on how lifetimes work.

Using boxes and references together is very common. For example:

```{rust}
fn add_one(x: &int) -> int {
    *x + 1
}

fn main() {
    let x = box 5i;

    println!("{}", add_one(&*x));
}
```

In this case, Rust knows that `x` is being 'borrowed' by the `add_one()`
function, and since it's only reading the value, allows it.

We can borrow `x` multiple times, as long as it's not simultaneous:

```{rust}
fn add_one(x: &int) -> int {
    *x + 1
}

fn main() {
    let x = box 5i;

    println!("{}", add_one(&*x));
    println!("{}", add_one(&*x));
    println!("{}", add_one(&*x));
}
```

Or as long as it's not a mutable borrow. This will error:

```{rust,ignore}
fn add_one(x: &mut int) -> int {
    *x + 1
}

fn main() {
    let x = box 5i;

    println!("{}", add_one(&*x)); // error: cannot borrow immutable dereference 
                                  // of `&`-pointer as mutable
}
```

Notice we changed the signature of `add_one()` to request a mutable reference.

## Best practices

Boxes are appropriate to use in two situations: Recursive data structures,
and occasionally, when returning data.

### Recursive data structures

Sometimes, you need a recursive data structure. The simplest is known as a
'cons list':


```{rust}
#[deriving(Show)]
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

fn main() {
    let list: List<int> = Cons(1, box Cons(2, box Cons(3, box Nil)));
    println!("{}", list);
}
```

This prints:

```{notrust,ignore}
Cons(1, box Cons(2, box Cons(3, box Nil)))
```

The reference to another `List` inside of the `Cons` enum variant must be a box,
because we don't know the length of the list. Because we don't know the length,
we don't know the size, and therefore, we need to heap allocate our list.

Working with recursive or other unknown-sized data structures is the primary
use-case for boxes.

### Returning data

This is important enough to have its own section entirely. The TL;DR is this:
you don't generally want to return pointers, even when you might in a language
like C or C++.

See [Returning Pointers](#returning-pointers) below for more.

# Rc and Arc

This part is coming soon.

## Best practices

This part is coming soon.

# Gc

The `Gc<T>` type exists for historical reasons, and is [still used
internally](https://github.com/rust-lang/rust/issues/7929) by the compiler.
It is not even a 'real' garbage collected type at the moment.

In the future, Rust may have a real garbage collected type, and so it
has not yet been removed for that reason.

## Best practices

There is currently no legitimate use case for the `Gc<T>` type.

# Raw Pointers

This part is coming soon.

## Best practices

This part is coming soon.

# Returning Pointers

In many languages with pointers, you'd return a pointer from a function
so as to avoid a copying a large data structure. For example:

```{rust}
struct BigStruct {
    one: int,
    two: int,
    // etc
    one_hundred: int,
}

fn foo(x: Box<BigStruct>) -> Box<BigStruct> {
    return box *x;
}

fn main() {
    let x = box BigStruct {
        one: 1,
        two: 2,
        one_hundred: 100,
    };

    let y = foo(x);
}
```

The idea is that by passing around a box, you're only copying a pointer, rather
than the hundred `int`s that make up the `BigStruct`.

This is an antipattern in Rust. Instead, write this:

```{rust}
struct BigStruct {
    one: int,
    two: int,
    // etc
    one_hundred: int,
}

fn foo(x: Box<BigStruct>) -> BigStruct {
    return *x;
}

fn main() {
    let x = box BigStruct {
        one: 1,
        two: 2,
        one_hundred: 100,
    };

    let y = box foo(x);
}
```

This gives you flexibility without sacrificing performance.

You may think that this gives us terrible performance: return a value and then
immediately box it up ?! Isn't that the worst of both worlds? Rust is smarter
than that. There is no copy in this code. main allocates enough room for the
`box , passes a pointer to that memory into foo as x, and then foo writes the
value straight into that pointer. This writes the return value directly into
the allocated box.

This is important enough that it bears repeating: pointers are not for
optimizing returning values from your code. Allow the caller to choose how they
want to use your output.

# Creating your own Pointers

This part is coming soon.

## Best practices

This part is coming soon.

# Cheat Sheet

Here's a quick rundown of Rust's pointer types:

| Type         | Name                | Summary                                                             |
|--------------|---------------------|---------------------------------------------------------------------|
| `&T`         | Reference           | Allows one or more references to read `T`                           |
| `&mut T`     | Mutable Reference   | Allows a single reference to read and write `T`                     |
| `Box<T>`     | Box                 | Heap allocated `T` with a single owner that may read and write `T`. |
| `Rc<T>`      | "arr cee" pointer   | Heap allocated `T` with many readers                                |
| `Arc<T>`     | Arc pointer         | Same as above, but safe sharing across threads                      |
| `*const T`   | Raw pointer         | Unsafe read access to `T`                                           |
| `*mut T`     | Mutable raw pointer | Unsafe read and write access to `T`                                 |

# Related resources

* [API documentation for Box](std/boxed/index.html)
* [Lifetimes guide](guide-lifetimes.html)
* [Cyclone paper on regions](http://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf), which inspired Rust's lifetime system
