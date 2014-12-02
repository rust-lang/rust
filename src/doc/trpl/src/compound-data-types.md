% Compound Data Types

Rust, like many programming languages, has a number of different data types
that are built-in. You've already done some simple work with integers and
strings, but next, let's talk about some more complicated ways of storing data.

## Tuples

The first compound data type we're going to talk about are called **tuple**s.
Tuples are an ordered list of a fixed size. Like this:

```rust
let x = (1i, "hello");
```

The parentheses and commas form this two-length tuple. Here's the same code, but
with the type annotated:

```rust
let x: (int, &str) = (1, "hello");
```

As you can see, the type of a tuple looks just like the tuple, but with each
position having a type name rather than the value. Careful readers will also
note that tuples are heterogeneous: we have an `int` and a `&str` in this tuple.
You haven't seen `&str` as a type before, and we'll discuss the details of
strings later. In systems programming languages, strings are a bit more complex
than in other languages. For now, just read `&str` as "a string slice," and
we'll learn more soon.

You can access the fields in a tuple through a **destructuring let**. Here's
an example:

```rust
let (x, y, z) = (1i, 2i, 3i);

println!("x is {}", x);
```

Remember before when I said the left-hand side of a `let` statement was more
powerful than just assigning a binding? Here we are. We can put a pattern on
the left-hand side of the `let`, and if it matches up to the right-hand side,
we can assign multiple bindings at once. In this case, `let` 'destructures,'
or 'breaks up,' the tuple, and assigns the bits to three bindings.

This pattern is very powerful, and we'll see it repeated more later.

There also a few things you can do with a tuple as a whole, without
destructuring. You can assign one tuple into another, if they have the same
arity and contained types.

```rust
let mut x = (1i, 2i);
let y = (2i, 3i);

x = y;
```

You can also check for equality with `==`. Again, this will only compile if the
tuples have the same type.

```rust
let x = (1i, 2i, 3i);
let y = (2i, 2i, 4i);

if x == y {
    println!("yes");
} else {
    println!("no");
}
```

This will print `no`, because some of the values aren't equal.

One other use of tuples is to return multiple values from a function:

```rust
fn next_two(x: int) -> (int, int) { (x + 1i, x + 2i) }

fn main() {
    let (x, y) = next_two(5i);
    println!("x, y = {}, {}", x, y);
}
```

Even though Rust functions can only return one value, a tuple _is_ one value,
that happens to be made up of two. You can also see in this example how you
can destructure a pattern returned by a function, as well.

Tuples are a very simple data structure, and so are not often what you want.
Let's move on to their bigger sibling, structs.

## Structs

A struct is another form of a 'record type,' just like a tuple. There's a
difference: structs give each element that they contain a name, called a
'field' or a 'member.' Check it out:

```rust
struct Point {
    x: int,
    y: int,
}

fn main() {
    let origin = Point { x: 0i, y: 0i };

    println!("The origin is at ({}, {})", origin.x, origin.y);
}
```

There's a lot going on here, so let's break it down. We declare a struct with
the `struct` keyword, and then with a name. By convention, structs begin with a
capital letter and are also camel cased: `PointInSpace`, not `Point_In_Space`.

We can create an instance of our struct via `let`, as usual, but we use a `key:
value` style syntax to set each field. The order doesn't need to be the same as
in the original declaration.

Finally, because fields have names, we can access the field through dot
notation: `origin.x`.

The values in structs are immutable, like other bindings in Rust. However, you
can use `mut` to make them mutable:

```{rust}
struct Point {
    x: int,
    y: int,
}

fn main() {
    let mut point = Point { x: 0i, y: 0i };

    point.x = 5;

    println!("The point is at ({}, {})", point.x, point.y);
}
```

This will print `The point is at (5, 0)`.

## Tuple Structs and Newtypes

Rust has another data type that's like a hybrid between a tuple and a struct,
called a **tuple struct**. Tuple structs do have a name, but their fields
don't:


```{rust}
struct Color(int, int, int);
struct Point(int, int, int);
```

These two will not be equal, even if they have the same values:

```{rust,ignore}
let black  = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

It is almost always better to use a struct than a tuple struct. We would write
`Color` and `Point` like this instead:

```{rust}
struct Color {
    red: int,
    blue: int,
    green: int,
}

struct Point {
    x: int,
    y: int,
    z: int,
}
```

Now, we have actual names, rather than positions. Good names are important,
and with a struct, we have actual names.

There _is_ one case when a tuple struct is very useful, though, and that's a
tuple struct with only one element. We call this a 'newtype,' because it lets
you create a new type that's a synonym for another one:

```{rust}
struct Inches(int);

let length = Inches(10);

let Inches(integer_length) = length;
println!("length is {} inches", integer_length);
```

As you can see here, you can extract the inner integer type through a
destructuring `let`.

## Enums

Finally, Rust has a "sum type", an **enum**. Enums are an incredibly useful
feature of Rust, and are used throughout the standard library. This is an enum
that is provided by the Rust standard library:

```{rust}
enum Ordering {
    Less,
    Equal,
    Greater,
}
```

An `Ordering` can only be _one_ of `Less`, `Equal`, or `Greater` at any given
time. Here's an example:

```{rust}
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    let ordering = cmp(x, y);

    if ordering == Less {
        println!("less");
    } else if ordering == Greater {
        println!("greater");
    } else if ordering == Equal {
        println!("equal");
    }
}
```

`cmp` is a function that compares two things, and returns an `Ordering`. We
return either `Less`, `Greater`, or `Equal`, depending on if the two values
are greater, less, or equal.

The `ordering` variable has the type `Ordering`, and so contains one of the
three values. We can then do a bunch of `if`/`else` comparisons to check
which one it is.

However, repeated `if`/`else` comparisons get quite tedious. Rust has a feature
that not only makes them nicer to read, but also makes sure that you never
miss a case. Before we get to that, though, let's talk about another kind of
enum: one with values.

This enum has two variants, one of which has a value:

```{rust}
enum OptionalInt {
    Value(int),
    Missing,
}
```

This enum represents an `int` that we may or may not have. In the `Missing`
case, we have no value, but in the `Value` case, we do. This enum is specific
to `int`s, though. We can make it usable by any type, but we haven't quite
gotten there yet!

You can also have any number of values in an enum:

```{rust}
enum OptionalColor {
    Color(int, int, int),
    Missing,
}
```

And you can also have something like this:

```{rust}
enum StringResult {
    StringOK(String),
    ErrorReason(String),
}
```
Where a `StringResult` is either an `StringOK`, with the result of a computation, or an
`ErrorReason` with a `String` explaining what caused the computation to fail. These kinds of
`enum`s are actually very useful and are even part of the standard library.

As you can see `enum`s with values are quite a powerful tool for data representation,
and can be even more useful when they're generic across types. But before we get to
generics, let's talk about how to use them with pattern matching, a tool that will
let us deconstruct this sum type (the type theory term for enums) in a very elegant
way and avoid all these messy `if`/`else`s.

