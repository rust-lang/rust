% The Rust Ownership Guide

One of Rust's most important features is its system of ownership. Other
important concepts like borrowing and lifetimes follow from the ownership
system too. A solid understanding of ownership is what separates an
intermediate Rust programmer from a new one.

At the same time, ownership is one of the more unique features that Rust has,
and so it may seem a bit strange at first. With a little bit of studying, the
ownership system will become second nature.

# Ownership

Virtually all programs make use of some kind of shared resource on your
computer. The most common of these is memory. In order to use a particular
resource, you are given a reference to that resource. It's through this
reference that you are able to utilize the underlying resource. As an example,
here is some Rust code which allocates some memory on the heap, and stores
an integer there:

```{rust}
fn main() {
    let x = box 5i;
}
```

The `box` keyword allocates memory on the heap, and places a `Box<int>` there.
`x` is then used to refer to that heap memory.

Each resource can have at most one **owner**. In this case, `x` is the owner of
the box. An 'owner' is able to deallocate, de-provision, or otherwise free the
particular resource in question. In the given example, since `x` owns the box,
when `x` goes out of scope, the Rust compiler sees that the owner is going
away, and therefore, frees the heap memory for us. This is the core of Rust's
strategy for memory management. Instead of having the programmer manage
deallocating resources, or having a garbage collector pick up after a
programmer, Rust's compiler is able to analyze the ownership of your resource
usage, and free the resource for you.

# Borrowing

Of course, it would be very limiting to only have a single possible pointer to
a resource. If you need additional pointers, Rust offers a second concept:
borrowing. Another reference may borrow the contents of a pointer which has
ownership. Here's an example:

```{rust}
fn add_one(i: &int) -> int {
    *i + 1
}

fn main() {
    let x = 5;
    println!("{}", add_one(&x));
}
```

The `&x` on the seventh line creates a reference to `x`, and the `add_one`
function takes a reference to an `int` as a parameter. In this case, the `i`
parameter is said to be **borrowing** the value.

What's the difference between being an owner and borrowing a reference? A major
difference is that when a reference goes out of scope, the resource is _not_
deallocated. The owner still exists, of course! If we did a deallocation when a
reference went out of scope, the owner would point to something invalid. There
are some restrictions placed upon when you may borrow a reference, but we'll
discuss those later in this guide.

# The borrow checker

It's important to understand that ownership and borrowing are both entirely
_compile time_ concepts in Rust. That is, the rules involving both are checked
entirely at compile time, so you pay no runtime penalty. The part of the
compiler that does this is called 'the borrow checker,' and it's a friend and
foe to all Rustaceans.

Your first interaction with the borrow checker probably resulted from an
error message that looks a bit like this:

```{notrust}
error: cannot borrow immutable local variable `x` as mutable
     let y = &mut x;
                  ^
```

or maybe

```{notrust}
 2:19 error: missing lifetime specifier [E0106]
 a_string: &str,
           ^~~~
```

or even

```{notrust}
34:29 error: cannot borrow `env` as mutable more than once at a time
        let e = &mut env;
                     ^~~
29:29 note: previous borrow of `env` occurs here; the mutable borrow prevents subsequent moves, borrows, or modification of `env` until the borrow ends
        let e = &mut env;
                     ^~~
6:6 note: previous borrow ends here

}
^
```

They indicate that the compiler believes that you're breaking the rules around
who and what can borrow, in what way, and when. But before we get into the
details around those rules, these error messages mention the last concept we
haven't discussed yet: lifetimes.

# Lifetimes

A **lifetime** is defined as 'the length of time that a reference exists.' Why
do we care about how long our references live?  Let's see some code:

```{rust,ignore}
struct SomeStruct {
    a_string: &str,
}
```

If we try to compile this code, we get an error:

```{ignore,notrust}
2:19 error: missing lifetime specifier [E0106]
a_string: &str,
          ^~~~
```

The error here is 'missing lifetime specifier'. Why do we need a specifier?
Well, consider this code:

```{rust,ignore}
struct SomeStruct {
    a_string: &str,
}

fn main() {
    let x = box SomeStruct { a_string: "A string" };
}
```

Here, `x` is a box with a `SomeStruct` inside. So we have this pointer to the
structure itself, but the structure _also_ contains a pointer to a `str`.  We
have no guarantee that the reference inside of `a_string` is valid for the
entire time that the reference to the `SomeStruct` itself. If it's not, then we
could have a valid reference to a `SomeStruct` which contained an invalid
reference in its `a_string`. That would lead to problems.

We need a way to tell the compiler that we expect `a_string` must be valid
as long as a reference to the struct itself is valid. We can do this with an
**explicit lifetime annotation**:

```{rust}
struct SomeStruct<'a> {
    a_string: &'a str,
}

fn main() {
    let x = box SomeStruct { a_string: "A string" };
}
```

This code compiles. We have two additions: a `<'a>` and a `&'a`. These two are
related. Changing `struct SomeStruct` to `struct SomeStruct<'a>` adds a
**lifetime parameter**. This parameter gives a name to the lifetime of a
reference to this struct.

Now that we've named the lifetime of a reference to the struct, we can use it
inside the definition. We change `&str` to `&'a str`, which says that
rather than having its own lifetime, `a_string` will have the lifetime of
`'a`, which is what we named the lifetime of the struct itself. Now,
we've tied the two together: we've told Rust that the lifetime of `a_string`
will last as long as the reference to the struct itself. Since they live for
the same amount of time, we'll never have an internal invalid reference, and so
Rust allows the code to compile.

# The rules of borrowing

# The rules of lifetime elision

# Shared ownership






















