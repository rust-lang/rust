- Feature Name: `match_vert_prefix
- Start Date: 2017-02-23
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This is a proposal for the rust grammar to support verts `|` *both*
at the beginning of the pattern and at the end. Consider the following
example:

```rust
use E::*;

enum E { A, B, C, D }

// This is valid Rust
match foo {
    A | B | C | D => (),
}

// This is an example of what this proposal should allow.
match foo {
    | A | B | C | D | => (),
}
```

# Motivation
[motivation]: #motivation

This is taking a feature which is nice about `F#` and allowing it by a
straightforward extension of the current rust language. After having used
this in `F#`, it seems limiting to not even support this at the language
level.

## `F#` Context

In `F#`, enumerations (called `unions`) are declared in the following fashion where
all of these are equivalent:

```F#
// Normal union
type IntOrBool = I of int | B of bool
// For consistency, have all lines look the same
type IntOrBool = 
   | I of int
   | B of bool
// Collapsing onto a single line is allowed
type IntOrBool = | I of int | B of bool
```

Their `match` statements adopt a similar style to this. Note that every `|` is aligned,
something which is not possible with current Rust:

```F#
match foo with
    | I -> ""
    | B -> ""
```

## Maximizing `|` alignment

In Rust, about the best we can do is align via the trailing edge:

```rust
use E::*;

enum E { A, B, C, D }

match foo {
    A |
    B |
    C |
    D => (),
//    ^ Inconsistently missing a `|`
}
```

Allowing this proposal would allow this example to take one of the following forms:

```rust
use E::*;

enum E { A, B, C, D }

match foo {
    A |
    B |
    C |
    D | => (),
//    ^ Gained consistency by having a vert.
}

match foo {
    | A
    | B
    | C
    | D => (),
//  ^ Gained *both* an alternative style and consistency.
}
```

## Flexibility in single line matches

It would allow these examples which are all equivalent:

```rust
use E::*;

enum E { A, B, C, D }

// Only trailing `|`.
match foo {
    A | B | C | D | => (),
}

// Only preceding `|`.
match foo {
    | A | B | C | D => (),
}

// Both preceding and trailing `|`.
match foo {
    | A | B | C | D | => (),
}

// Neither preceding nor trailing `|`.
match foo {
    A | B | C | D => (),
}
```

> There should be no ambiguity about what each of these mean. Preference
between these should just come down to a choice of style.

## Benefits to macros

This benefits macros. Needs filling in.

## Multiple branches

All of these matches are equivalent, each written in a different style:

```rust
use E::*;

enum E { A, B, C, D }

match foo {
    A | B => println!("Give me A | B!"),
    C | D => println!("Give me C | D!"),
}

match foo {
    | A | B => println!("Give me A | B!"),
    | C | D => println!("Give me C | D!"),
}

match foo {
    | A
    | B => println!("Give me A | B!"),
    | C
    | D => println!("Give me C | D!"),
}

match foo {
    A | B =>
        println!("Give me A | B!"),
    C | D =>
        println!("Give me C | D!"),
}

match foo {
    A |
    B | => println!("Give me A | B!"),
    C |
    D | => println!("Give me C | D!"),
}
```

## Comparing misalignment

```rust
use E::*;

enum E { A, B, C }

match foo {
    | A
    | B => {},
    | C => {}
//  ^ Following the style above, a `|` could be placed before the first
// element of every branch.

match value {
    | A
    | B => {},
    C => {}
//  ^ Including a `|` for the `A` but not for the `C` seems inconsistent
// but hardly invalid. Branches *always* follow the `=>`. Not something
// a *grammar* should be greatly concerned about.
}
```

# Detailed design
[design]: #detailed-design

I don't know about the implementation but the grammar could be updated so that
an optional `|` is allowed at the beginning. Nothing else in the grammar should
need updating.

```text
// Before
match_pat : pat [ '|' pat ] * [ "if" expr ] ? ;
// After
match_pat : '|' ? pat [ '|' pat ] * '|' ? [ "if" expr ] ? ;
```

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Adding examples for this are straightforward. You just include an example pointing
out that leading verts are allowed. Simple examples such as below should be easy to
add to all different resources.

```rust
enum Cat {
    Burmese,
    Munchkin,
    Siamese,
}

enum Dog {
    Dachshund,
    Poodle,
    PitBull,
}

fn main() {
    let cat = Cat::MunchKin;
    let dog = Dog::Poodle;

    match cat {
        Cat::Burmese => "Burmese",
        // Can do alternatives with a `|`.
        Cat::MunchKin | Cat::Siamese => "Not burmese",
    }

    match dog {
        | Dog::Dachshund => "Dachshund",
        // Leading `|` is allowed.
        | Dog::Poodle
        | Dog::PitBull => "Not a dachshund",
    }
}
```

# Drawbacks
[drawbacks]: #drawbacks

N/A

# Alternatives
[alternatives]: #alternatives

N/A

# Unresolved questions
[unresolved]: #unresolved-questions

N/A
