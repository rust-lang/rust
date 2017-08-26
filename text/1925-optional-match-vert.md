- Feature Name: `match_vert_prefix`
- Start Date: 2017-02-23
- RFC PR: https://github.com/rust-lang/rfcs/pull/1925
- Rust Issue: https://github.com/rust-lang/rust/issues/44101

# Summary
[summary]: #summary

This is a proposal for the rust grammar to support a vert `|` at the
beginning of the pattern. Consider the following example:

```rust
use E::*;

enum E { A, B, C, D }

// This is valid Rust
match foo {
    A | B | C | D => (),
}

// This is an example of what this proposal should allow.
match foo {
    | A | B | C | D => (),
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

In Rust, about the best we can do is an inconsistent alignment with one of the
following two options:

```rust
use E::*;

enum E { A, B, C, D }

match foo {
//  |
//  V Inconsistently missing a `|`.
      A
    | B
    | C
    | D => (),
}

match foo {
    A |
    B |
    C |
    D => (),
//    ^ Also inconsistent but since this is the last in the sequence, not having 
//    | a followup vert could be considered sensible given that no more follow.
}
```

This proposal would allow the example to have the following form:

```rust
use E::*;

enum E { A, B, C, D }

match foo {
    | A
    | B
    | C
    | D => (),
//  ^ Gained consistency by having a matching vert.
}
```

## Flexibility in single line matches

It would allow these examples which are all equivalent:

```rust
use E::*;

enum E { A, B, C, D }

// A preceding vert
match foo {
    | A | B | C | D => (),
}

// A match as is currently allowed
match foo {
    A | B | C | D => (),
}
```

> There should be no ambiguity about what either of these means. Preference
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
match_pat : '|' ? pat [ '|' pat ] * [ "if" expr ] ? ;
```

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Adding examples for this are straightforward. You just include an example pointing
out that leading verts are allowed. Simple examples such as below should be easy
to add to all different resources.

```rust
use Letter::*;

enum Letter {
    A,
    B,
    C,
    D,
}

fn main() {
    let a = Letter::A;
    let b = Letter::B;
    let c = Letter::C;
    let d = Letter::D;

    match a {
        A => "A",
        // Can do alternatives with a `|`.
        B | C | D => "B, C, or D",
    }

    match b {
        | A => "A",
        // Leading `|` is allowed.
        | B
        | C
        | D => "B, C, or D",
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
