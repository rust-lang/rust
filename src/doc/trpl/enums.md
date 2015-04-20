% Enums

Rust has a ‘sum type’, an `enum`. Enums are an incredibly useful feature of
Rust, and are used throughout the standard library. An `enum` is a type which
relates a set of alternates to a specific name. For example, below we define
`Character` to be either a `Digit` or something else.

```rust
enum Character {
    Digit(i32),
    Other,
}
```

Most types are allowed as the variant components of an `enum`. Here are some
examples:

```rust
struct Empty;
struct Color(i32, i32, i32);
struct Length(i32);
struct Stats { Health: i32, Mana: i32, Attack: i32, Defense: i32 }
struct HeightDatabase(Vec<i32>);
```

You see that, depending on its type, an `enum` variant may or may not hold data.
In `Character`, for instance, `Digit` gives a meaningful name for an `i32`
value, where `Other` is only a name. However, the fact that they represent
distinct categories of `Character` is a very useful property.

The variants of an `enum` by default are not comparable with equality operators
(`==`, `!=`), have no ordering (`<`, `>=`, etc.), and do not support other
binary operations such as `*` and `+`. As such, the following code is invalid
for the example `Character` type:

```rust,ignore
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

We use the `::` syntax to use the name of each variant: They’re scoped by the name
of the `enum` itself. This allows both of these to work:

```rust,ignore
Character::Digit(10);
Hand::Digit;
```

Both variants are named `Digit`, but since they’re scoped to the `enum` name,

Not supporting these operations may seem rather limiting, but it’s a limitation
which we can overcome. There are two ways: by implementing equality ourselves,
or by pattern matching variants with [`match`][match] expressions, which you’ll
learn in the next section. We don’t know enough about Rust to implement
equality yet, but we’ll find out in the [`traits`][traits] section.

[match]: match.html
[traits]: traits.html
