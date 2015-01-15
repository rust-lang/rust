% Compound Data Types

Rust, like many programming languages, has a number of different data types
that are built-in. You've already done some simple work with integers and
strings, but next, let's talk about some more complicated ways of storing data.

## Tuples

The first compound data type we're going to talk about are called *tuples*.
Tuples are an ordered list of a fixed size. Like this:

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
arity and contained types.

```rust
let mut x = (1, 2); // x: (i32, i32)
let y = (2, 3);     // y: (i32, i32)

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
let black  = Color(0, 0, 0);
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
tuple struct with only one element. We call this a *newtype*, because it lets
you create a new type that's a synonym for another one:

```{rust}
struct Inches(i32);

let length = Inches(10);

let Inches(integer_length) = length;
println!("length is {} inches", integer_length);
```

As you can see here, you can extract the inner integer type through a
destructuring `let`.

## Enums

Finally, Rust has a "sum type", an *enum*. Enums are an incredibly useful
feature of Rust, and are used throughout the standard library. An `enum` is
a type which ties a set of alternates to a specific name. For example, below
we define `Character` to be either a `Digit` or something else. These
can be used via their fully scoped names: `Character::Other`.

```rust
enum Character {
    Digit(i32),
    Other,
}
```

An `enum` variant, can be defined as most normal types. Below some example types
have been listed which also would be allowed in an `enum`.

```rust
struct Empty;
struct Color(i32, i32, i32);
struct Length(i32);
struct Status { Health: i32, Mana: i32, Attack: i32, Defense: i32 }
struct HeightDatabase(Vec<i32>);
```

So you see that depending on the sub-datastructure, the `enum` variant, same as a
struct variant, may or may not hold data. That is, in `Character`, `Digit` is a name
tied to an `i32` where `Other` is just a name. However, the fact that they are distinct
makes this very useful.

As with structures, enums don't by default have access to operators such as
compare ( `==` and `!=`), binary operations (`*` and `+`), and order
(`<` and `>=`). As such, using the previous `Character` type, the
following code is invalid:

```{rust, ignore}
// These assignments both succeed
let ten  = Character::Digit(10);
let four = Character::Digit(4); 

// Error: `*` is not implemented for type `Character`
let forty = ten * four;

// Error: `<` is not implemented for type `Character`
let four_is_smaller = four < ten;

// Error: `==` is not implemented for type `Character`
let four_equals_ten = four == ten;
```

This may seem rather limiting, particularly equality being invalid; in
many cases however, it's unnecessary. Rust provides the `match` keyword,
which will be examined in more detail in the next section, which allows
better and easier branch control than a series of `if`/`else` statements
would. Here, we'll briefly utilize it to avoid some complicated
alternatives.

In spite of not having equality, the match below is able to deduce the type
of the object and take the corresponding branch. It can even retrieve the
number from inside the structure. This is the typical way an `enum` is
used.

```rust
enum Character {
    Digit(i32),
    Other,
}

let nine = Character::Digit(9i32);
let not_a_digit = Other;

match nine {
    Character::Digit(num) => println!("Got the digit: {:?}", num),
    Character::Other      => println!("Got the something else"),
}

match not_a_digit {
    Character::Digit(num) => println!("Got the digit: {:?}", num),
    Character::Other      => println!("Got the something else"),
}
```

As this is very verbose, it can be shortened using the `use` declaration.
`use` must precede everything so we put it at the top.

```rust
use Character::Digit;
use Character::Other;

enum Character {
    Digit(i32),
    Other,
}

let nine = Digit(9i32);
let not_a_digit = Other;

match nine {
    Digit(num) => println!("Got the digit: {:?}", num),
    Other      => println!("Got the something else"),
}

match not_a_digit {
    Digit(num) => println!("Got the digit: {:?}", num),
    Other      => println!("Got the something else"),
}
```

Importing variants can be convenient, but can also cause name conflicts, so do this
with caution. It's considered good style to rarely import variants for this reason.

As you can see, `enum`s with values are quite a powerful tool for data representation,
and can be even more useful when they're generic across types. Before we get to generics,
though, let's go into more detail about pattern matching which uses that `match` tool
we just saw.
