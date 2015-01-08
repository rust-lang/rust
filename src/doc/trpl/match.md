% Match

Often, a simple `if`/`else` isn't enough, because you have more than two
possible options. Also, `else` conditions can get incredibly complicated, so
what's the solution?

Rust has a keyword, `match`, that allows you to replace complicated `if`/`else`
groupings with something more powerful. Check it out:

```{rust}
let x = 5;

match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    4 => println!("four"),
    5 => println!("five"),
    _ => println!("something else"),
}
```

`match` takes an expression and then branches based on its value. Each 'arm' of
the branch is of the form `val => expression`. When the value matches, that arm's
expression will be evaluated. It's called `match` because of the term 'pattern
matching', which `match` is an implementation of.

So what's the big advantage here? Well, there are a few. First of all, `match`
enforces 'exhaustiveness checking'. Do you see that last arm, the one with the
underscore (`_`)? If we remove that arm, Rust will give us an error:

```text
error: non-exhaustive patterns: `_` not covered
```

In other words, Rust is trying to tell us we forgot a value. Because `x` is an
integer, Rust knows that it can have a number of different values – for example,
`6`. Without the `_`, however, there is no arm that could match, and so Rust refuses
to compile. `_` acts like a 'catch-all arm'. If none of the other arms match,
the arm with `_` will, and since we have this catch-all arm, we now have an arm
for every possible value of `x`, and so our program will compile successfully.

`match` statements also destructure enums, as well. Remember this code from the
section on enums?

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

    let ordering = cmp(x, y);

    if ordering == Ordering::Less {
        println!("less");
    } else if ordering == Ordering::Greater {
        println!("greater");
    } else if ordering == Ordering::Equal {
        println!("equal");
    }
}
```

We can re-write this as a `match`:

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

    match cmp(x, y) {
        Ordering::Less    => println!("less"),
        Ordering::Greater => println!("greater"),
        Ordering::Equal   => println!("equal"),
    }
}
```

This version has way less noise, and it also checks exhaustively to make sure
that we have covered all possible variants of `Ordering`. With our `if`/`else`
version, if we had forgotten the `Greater` case, for example, our program would
have happily compiled. If we forget in the `match`, it will not. Rust helps us
make sure to cover all of our bases.

`match` expressions also allow us to get the values contained in an `enum`
(also known as destructuring) as follows:

```{rust}
enum OptionalInt {
    Value(i32),
    Missing,
}

fn main() {
    let x = OptionalInt::Value(5);
    let y = OptionalInt::Missing;

    match x {
        OptionalInt::Value(n) => println!("x is {}", n),
        OptionalInt::Missing  => println!("x is missing!"),
    }

    match y {
        OptionalInt::Value(n) => println!("y is {}", n),
        OptionalInt::Missing  => println!("y is missing!"),
    }
}
```

That is how you can get and use the values contained in `enum`s.
It can also allow us to handle errors or unexpected computations; for example, a
function that is not guaranteed to be able to compute a result (an `i32` here)
could return an `OptionalInt`, and we would handle that value with a `match`.
As you can see, `enum` and `match` used together are quite useful!

`match` is also an expression, which means we can use it on the right-hand
side of a `let` binding or directly where an expression is used. We could
also implement the previous example like this:

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

    println!("{}", match cmp(x, y) {
        Ordering::Less    => "less",
        Ordering::Greater => "greater",
        Ordering::Equal   => "equal",
    });
}
```

Sometimes, it's a nice pattern.
