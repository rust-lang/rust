% Match

Often, a simple `if`/`else` isn't enough, because you have more than two
possible options. And `else` conditions can get incredibly complicated. So
what's the solution?

Rust has a keyword, `match`, that allows you to replace complicated `if`/`else`
groupings with something more powerful. Check it out:

```{rust}
let x = 5i;

match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    4 => println!("four"),
    5 => println!("five"),
    _ => println!("something else"),
}
```

`match` takes an expression, and then branches based on its value. Each 'arm' of
the branch is of the form `val => expression`. When the value matches, that arm's
expression will be evaluated. It's called `match` because of the term 'pattern
matching,' which `match` is an implementation of.

So what's the big advantage here? Well, there are a few. First of all, `match`
enforces 'exhaustiveness checking.' Do you see that last arm, the one with the
underscore (`_`)? If we remove that arm, Rust will give us an error:

```{ignore,notrust}
error: non-exhaustive patterns: `_` not covered
```

In other words, Rust is trying to tell us we forgot a value. Because `x` is an
integer, Rust knows that it can have a number of different values. For example,
`6i`. But without the `_`, there is no arm that could match, and so Rust refuses
to compile. `_` is sort of like a catch-all arm. If none of the other arms match,
the arm with `_` will. And since we have this catch-all arm, we now have an arm
for every possible value of `x`, and so our program will now compile.

`match` statements also destructure enums, as well. Remember this code from the
section on enums?

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

We can re-write this as a `match`:

```{rust}
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    match cmp(x, y) {
        Less    => println!("less"),
        Greater => println!("greater"),
        Equal   => println!("equal"),
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
    Value(int),
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
It can also allow us to treat errors or unexpected computations, for example, a
function that is not guaranteed to be able to compute a result (an `int` here),
could return an `OptionalInt`, and we would handle that value with a `match`.
As you can see, `enum` and `match` used together are quite useful!

`match` is also an expression, which means we can use it on the right
hand side of a `let` binding or directly where an expression is
used. We could also implement the previous line like this:

```{rust}
fn cmp(a: int, b: int) -> Ordering {
    if a < b { Less }
    else if a > b { Greater }
    else { Equal }
}

fn main() {
    let x = 5i;
    let y = 10i;

    println!("{}", match cmp(x, y) {
        Less    => "less",
        Greater => "greater",
        Equal   => "equal",
    });
}
```

Sometimes, it's a nice pattern.
