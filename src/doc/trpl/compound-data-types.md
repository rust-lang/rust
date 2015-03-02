% Compound Data Types

Rust, like many programming languages, has a number of different data types
that are built-in. You've already done some simple work with integers and
strings, but next, let's talk about some more complicated ways of storing data.

## Tuples

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

Tuples are a very simple data structure, and so are not often what you want.
Let's move on to their bigger sibling, structs.

## Structs

A struct is another form of a *record type*, just like a tuple. There's a
difference: structs give each element that they contain a name, called a
*field* or a *member*. Check it out:

```rust
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let origin = Point { x: 0, y: 0 }; // origin: Point

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

The values in structs are immutable by default, like other bindings in Rust.
Use `mut` to make them mutable:

```{rust}
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut point = Point { x: 0, y: 0 };

    point.x = 5;

    println!("The point is at ({}, {})", point.x, point.y);
}
```

This will print `The point is at (5, 0)`.

## Tuple Structs and Newtypes

Rust has another data type that's like a hybrid between a tuple and a struct,
called a *tuple struct*. Tuple structs do have a name, but their fields don't:


```{rust}
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);
```

These two will not be equal, even if they have the same values:

```{rust}
# struct Color(i32, i32, i32);
# struct Point(i32, i32, i32);
let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

It is almost always better to use a struct than a tuple struct. We would write
`Color` and `Point` like this instead:

```{rust}
struct Color {
    red: i32,
    blue: i32,
    green: i32,
}

struct Point {
    x: i32,
    y: i32,
    z: i32,
}
```

Now, we have actual names, rather than positions. Good names are important,
and with a struct, we have actual names.

There _is_ one case when a tuple struct is very useful, though, and that's a
tuple struct with only one element. We call this the *newtype* pattern, because
it allows you to create a new type, distinct from that of its contained value
and expressing its own semantic meaning:

```{rust}
struct Inches(i32);

let length = Inches(10);

let Inches(integer_length) = length;
println!("length is {} inches", integer_length);
```

As you can see here, you can extract the inner integer type through a
destructuring `let`, as we discussed previously in 'tuples.' In this case, the
`let Inches(integer_length)` assigns `10` to `integer_length`.

## Enums

Finally, Rust has a "sum type", an *enum*. Enums are an incredibly useful
feature of Rust, and are used throughout the standard library. An `enum` is
a type which relates a set of alternates to a specific name. For example, below
we define `Character` to be either a `Digit` or something else. These
can be used via their fully scoped names: `Character::Other` (more about `::`
below).

```rust
enum Character {
    Digit(i32),
    Other,
}
```

Most normal types are allowed as the variant components of an `enum`. Here are
some examples:

```rust
struct Empty;
struct Color(i32, i32, i32);
struct Length(i32);
struct Status { Health: i32, Mana: i32, Attack: i32, Defense: i32 }
struct HeightDatabase(Vec<i32>);
```

You see that, depending on its type, an `enum` variant may or may not hold data.
In `Character`, for instance, `Digit` gives a meaningful name for an `i32`
value, where `Other` is only a name. However, the fact that they represent
distinct categories of `Character` is a very useful property.

As with structures, the variants of an enum by default are not comparable with
equality operators (`==`, `!=`), have no ordering (`<`, `>=`, etc.), and do not
support other binary operations such as `*` and `+`. As such, the following code
is invalid for the example `Character` type:

```{rust,ignore}
// These assignments both succeed
let ten  = Character::Digit(10);
let four = Character::Digit(4);

// Error: `*` is not implemented for type `Character`
let forty = ten * four;

// Error: `<=` is not implemented for type `Character`
let four_is_smaller = four <= ten;

// Error: `==` is not implemented for type `Character`
let four_equals_ten = four == ten;
```

This may seem rather limiting, but it's a limitation which we can overcome.
There are two ways: by implementing equality ourselves, or by pattern matching
variants with [`match`][match] expressions, which you'll learn in the next
chapter. We don't know enough about Rust to implement equality yet, but we can
use the `Ordering` enum from the standard library, which does:

```
enum Ordering {
    Less,
    Equal,
    Greater,
}
```

Because `Ordering` has already been defined for us, we will import it with the
`use` keyword. Here's an example of how it is used:

```{rust}
use std::cmp::Ordering;

fn cmp(a: i32, b: i32) -> Ordering {
    if a < b { Ordering::Less }
    else if a > b { Ordering::Greater }
    else { Ordering::Equal }
}

fn main() {
    let x = 5;
    let y = 10;

    let ordering = cmp(x, y); // ordering: Ordering

    if ordering == Ordering::Less {
        println!("less");
    } else if ordering == Ordering::Greater {
        println!("greater");
    } else if ordering == Ordering::Equal {
        println!("equal");
    }
}
```

The `::` symbol is used to indicate a namespace. In this case, `Ordering` lives
in the `cmp` submodule of the `std` module. We'll talk more about modules later
in the guide. For now, all you need to know is that you can `use` things from
the standard library if you need them.

Okay, let's talk about the actual code in the example. `cmp` is a function that
compares two things, and returns an `Ordering`. We return either
`Ordering::Less`, `Ordering::Greater`, or `Ordering::Equal`, depending on
whether the first value is less than, greater than, or equal to the second. Note
that each variant of the `enum` is namespaced under the `enum` itself: it's
`Ordering::Greater`, not `Greater`.

The `ordering` variable has the type `Ordering`, and so contains one of the
three values. We then do a bunch of `if`/`else` comparisons to check which
one it is.

This `Ordering::Greater` notation is too long. Let's use another form of `use`
to import the `enum` variants instead. This will avoid full scoping:

```{rust}
use std::cmp::Ordering::{self, Equal, Less, Greater};

fn cmp(a: i32, b: i32) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5;
    let y = 10;

    let ordering = cmp(x, y); // ordering: Ordering

    if ordering == Less { println!("less"); }
    else if ordering == Greater { println!("greater"); }
    else if ordering == Equal { println!("equal"); }
}
```

Importing variants is convenient and compact, but can also cause name conflicts,
so do this with caution. For this reason, it's normally considered better style
to `use` an enum rather than its variants directly.

As you can see, `enum`s are quite a powerful tool for data representation, and
are even more useful when they're [generic][generics] across types. Before we
get to generics, though, let's talk about how to use enums with pattern
matching, a tool that will let us deconstruct sum types (the type theory term
for enums) like `Ordering` in a very elegant way that avoids all these messy
and brittle `if`/`else`s.


[arity]: ./glossary.html#arity
[match]: ./match.html
[game]: ./guessing-game.html#comparing-guesses
[generics]: ./generics.html
