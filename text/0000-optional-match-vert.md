- Feature Name: `match_vert_prefix
- Start Date: 2017-02-23
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

See how in the following, all the `|` bars are mostly aligned except the last one:

```rust

use E::*;

enum E {
    A,
    B,
    C,
    D,
}

fn main() {
    match A {
        A |
        B |
        C |
        D => (),
    }
}
```

I'd propose it be allowed at the beginning of the pattern as well enabling something like this:

```rust
use E::*;

enum E {
    A,
    B,
    C,
    D,
}

fn main() {
    match A {
    | A
    | B
    | C
    | D => (),
    }
}
```

# Motivation
[motivation]: #motivation

This is taking a feature which is nice about `F#` and allowing it by a straightforward
extension of the current rust language.

Also, this appears to be the official style for F# matches and it has grown on me a lot.
It highlights the matches and doesn't require as much deeper nesting. After getting used to the
F# style, the inability to do this is rust seems a bit limiting.

## F# Context

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

Their `match` statements adopt a similar style to this:

```F#
match foo with
| I -> ""
| B -> ""
```

In Rust, these would look like this:

```rust
enum IntOrBool {
    I(i32),
    B(bool),
}

// Currently a preceding vert is disallowed
match foo {
| I => "",
| B => "",
}
```
The appealing feature about this is that this style allows `match` semantics without
requiring the double nesting of a typical `match`.

## Example A

All of these matches are equivalent.

```rust
enum E {
    A,
    B,
    C,
    D,
}

fn main() {
    match A {
        A | B => println!("Give me A | B!"),
        C | D => println!("Give me C | D!"),
    }

    match A {
    | A | B => println!("Give me A | B!"),
    | C | D => println!("Give me C | D!"),
    }

    match A {
    | A
    | B => println!("Give me A | B!"),
    | C
    | D => println!("Give me C | D!"),
    }

    match A {
        A | B =>
            println!("Give me A | B!"),
        C | D =>
            println!("Give me C | D!"),
    }
}
```

## Example B

```rust
enum E { A, B, C }

fn main() {
    use E::*;
    let value = A;

    match value {
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
// to be greatly concerned about.
    }
}
```

## Example C

A more thorough example is included below. Note how the bottom example how at most,
only tabs in twice from the start of the match. In contrast, the top tabs in four times.

```rust
struct FavoriteBook {
    author: &'static str,
    title: &'static str,
    date: u64
}

// Full name and surname. 
enum Franks { Alice, Brenda, Charles, Dave, Steve }
enum Sawyer { Tom, Sid, May }

enum Name {
    Franks(Franks),
    Sawyer(Sawyer),
}

fn main() {
    let name = Name::Sawyer(Sawyer::Tom);

    // Here is the first match in a typical rust style
    match name {
        Name::Franks(name) =>
            match name {
                Franks::Alice |
                Franks::Brenda |
                Franks::Dave => FavoriteBook {
                    author: "alice berkley",
                    title: "Name of a popular book",
                    date: 1982,
                },
                Franks::Charles |
                Franks::Steve => FavoriteBook {
                    author: "fred marko",
                    title: "We'll use a different name here",
                    date: 1960,
                },
			},
        Name::Sawyer(name) =>
            match name {
                Sawyer::Tom => FavoriteBook {
                    author: "another name",
                    title: "Again we change it",
                    date: 1999,
                },
                Sawyer::Sid |
                Sawyer::May => FavoriteBook {
                    author: "again another name",
                    title: "here is a different title",
                    date: 1972,
                },
            }
    };

    // An alternate rust style might look something like this:
    match name {
    | Name::Franks(name) =>
        match name {
        | Franks::Alice
        | Franks::Brenda
        | Franks::Dave => FavoriteBook {
            author: "alice berkley",
            title: "Name of a popular book",
            date: 1982
        },
        | Franks::Charles
        | Franks::Steve => FavoriteBook {
            author: "fred marko",
            title: "We'll use a different name here",
            date: 1960
        },
        }
    | Name::Sawyer(name) =>
        match name {
        | Sawyer::Tom => FavoriteBook {
            author: "another name",
            title: "Again we change it",
            date: 1999
        },
        | Sawyer::Sid
        | Sawyer::May => FavoriteBook {
            author: "again another name",
            title: "here is a different title",
            date: 1972
        },
        }
    };
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

Adding examples for this are straightforward. You just include an example pointing out that
leading verts are allowed. That style could also be shown as well if desired. Simple
examples such as below should be easy to add to all different resources.

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

Nesting braces without nesting gets a little weird. This doesn't seem problematic but
stylistically, it might just seem quirky. `F#` doesn't have this problem because they use
whitespace for nesting I believe.

```rust
struct S {msg: &'static str }
enum E { A, B, C }

fn main() {
    let e = E::A;

    match e {
    | E::A => S {
        msg: "A",
    },
    | E::B => S {
        msg: "B",
    },
    | E::C => S {
        msg: "C",
    },
    }
//  ^ Braces all hit the same level.
//    This example may seem trivial but example c also showcased the exact same thing.
```

# Alternatives
[alternatives]: #alternatives

N/A

# Unresolved questions
[unresolved]: #unresolved-questions

N/A
