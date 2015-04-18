% Move Semantics

An important aspect of [ownership][ownership] is ‘move semantics’. Move
semantics control how and when ownership is transferred between bindings.

[ownership]: ownership.html

For example, consider a type like `Vec<T>`, which owns its contents:

```rust
let v = vec![1, 2, 3];
```

I can assign this vector to another binding:

```rust
let v = vec![1, 2, 3];

let v2 = v;
```

But, if we try to use `v` afterwards, we get an error:

```rust,ignore
let v = vec![1, 2, 3];

let v2 = v;

println!("v[0] is: {}", v[0]);
```

It looks like this:

```text
error: use of moved value: `v`
println!("v[0] is: {}", v[0]);
                        ^
```

A similar thing happens if we define a function which takes ownership, and
try to use something after we’ve passed it as an argument:

```rust,ignore
fn take(v: Vec<i32>) {
    // what happens here isn’t important.
}

let v = vec![1, 2, 3];

take(v);

println!("v[0] is: {}", v[0]);
```

Same error: “use of moved value.” When we transfer ownership to something else,
we say that we’ve ‘moved’ the thing we refer to. You don’t need some sort of
special annotation here, it’s the default thing that Rust does.

# The details

The reason that we cannot use a binding after we’ve moved it is subtle, but
important. When we write code like this:

```rust
let v = vec![1, 2, 3];

let v2 = v;
```

The first line creates some data for the vector on the stack, `v`. The vector’s
data, however, is stored on the heap, and so it contains a pointer to that
data. When we move `v` to `v2`, it creates a copy of that data, for `v2`. Which
would mean two pointers to the contents of the vector on the heap. That would
be a problem: it would violate Rust’s safety guarantees by introducing a data
race. Therefore, Rust forbids using `v` after we’ve done the move.

It’s also important to note that optimizations may remove the actual copy of
the bytes, depending on circumstances. So it may not be as inefficient as it
initially seems.

# `Copy` types

We’ve established that when ownership is transferred to another binding, you
cannot use the original binding. However, there’s a [trait][traits] that changes this
behavior, and it’s called `Copy`. We haven’t discussed traits yet, but for now,
you can think of them as an annotation to a particular type that adds extra
behavior. For example:

```rust
let v = 1;

let v2 = v;

println!("v is: {}", v);
```

In this case, `v` is an `i32`, which implements the `Copy` trait. This means
that, just like a move, when we assign `v` to `v2`, a copy of the data is made.
But, unlike a move, we can still use `v` afterward. This is because an `i32`
has no pointers to data somewhere else, copying it is a full copy.

We will discuss how to make your own types `Copy` in the [traits][traits]
section.

[traits]: traits.html
