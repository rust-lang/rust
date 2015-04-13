% Enums

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

[match]: ./match.html
[generics]: ./generics.html
