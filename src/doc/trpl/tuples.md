% Tuples

The first compound data type we're going to talk about is called the *tuple*.
A tuple is an ordered list of fixed size. Like this:

```rust
let x = (1, "hello");
```

The parentheses and commas form this two-length tuple. Here's the same code, but
with the type annotated:

```rust
let x: (i32, &str) = (1, "hello");
```

As you can see, the type of a tuple looks just like the tuple, but with each
position having a type name rather than the value. Careful readers will also
note that tuples are heterogeneous: we have an `i32` and a `&str` in this tuple.
You have briefly seen `&str` used as a type before, and we'll discuss the
details of strings later. In systems programming languages, strings are a bit
more complex than in other languages. For now, just read `&str` as a *string
slice*, and we'll learn more soon.

You can access the fields in a tuple through a *destructuring let*. Here's
an example:

```rust
let (x, y, z) = (1, 2, 3);

println!("x is {}", x);
```

Remember before when I said the left-hand side of a `let` statement was more
powerful than just assigning a binding? Here we are. We can put a pattern on
the left-hand side of the `let`, and if it matches up to the right-hand side,
we can assign multiple bindings at once. In this case, `let` "destructures,"
or "breaks up," the tuple, and assigns the bits to three bindings.

This pattern is very powerful, and we'll see it repeated more later.

There are also a few things you can do with a tuple as a whole, without
destructuring. You can assign one tuple into another, if they have the same
contained types and [arity]. Tuples have the same arity when they have the same
length.

```rust
let mut x = (1, 2); // x: (i32, i32)
let y = (2, 3); // y: (i32, i32)

x = y;
```

You can also check for equality with `==`. Again, this will only compile if the
tuples have the same type.

```rust
let x = (1, 2, 3);
let y = (2, 2, 4);

if x == y {
    println!("yes");
} else {
    println!("no");
}
```

This will print `no`, because some of the values aren't equal.

Note that the order of the values is considered when checking for equality,
so the following example will also print `no`.

```rust
let x = (1, 2, 3);
let y = (2, 1, 3);

if x == y {
    println!("yes");
} else {
    println!("no");
}
```

One other use of tuples is to return multiple values from a function:

```rust
fn next_two(x: i32) -> (i32, i32) { (x + 1, x + 2) }

fn main() {
    let (x, y) = next_two(5);
    println!("x, y = {}, {}", x, y);
}
```

Even though Rust functions can only return one value, a tuple *is* one value,
that happens to be made up of more than one value. You can also see in this
example how you can destructure a pattern returned by a function, as well.
