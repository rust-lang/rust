% Match

Often, a simple [`if`][if]/`else` isn’t enough, because you have more than two
possible options. Also, conditions can get quite complex. Rust
has a keyword, `match`, that allows you to replace complicated `if`/`else`
groupings with something more powerful. Check it out:

```rust
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

[if]: if.html

`match` takes an expression and then branches based on its value. Each ‘arm’ of
the branch is of the form `val => expression`. When the value matches, that arm’s
expression will be evaluated. It’s called `match` because of the term ‘pattern
matching’, which `match` is an implementation of. There’s an [entire section on
patterns][patterns] that covers all the patterns that are possible here.

[patterns]: patterns.html

So what’s the big advantage? Well, there are a few. First of all, `match`
enforces ‘exhaustiveness checking’. Do you see that last arm, the one with the
underscore (`_`)? If we remove that arm, Rust will give us an error:

```text
error: non-exhaustive patterns: `_` not covered
```

In other words, Rust is trying to tell us we forgot a value. Because `x` is an
integer, Rust knows that it can have a number of different values – for
example, `6`. Without the `_`, however, there is no arm that could match, and
so Rust refuses to compile the code. `_` acts like a ‘catch-all arm’. If none
of the other arms match, the arm with `_` will, and since we have this
catch-all arm, we now have an arm for every possible value of `x`, and so our
program will compile successfully.

`match` is also an expression, which means we can use it on the right-hand
side of a `let` binding or directly where an expression is used:

```rust
let x = 5;

let numer = match x {
    1 => "one",
    2 => "two",
    3 => "three",
    4 => "four",
    5 => "five",
    _ => "something else",
};
```

Sometimes it’s a nice way of converting something from one type to another.
