- Feature Name: macro_2_0
- Start Date: 2016-04-17
- RFC PR: [1584](https://github.com/rust-lang/rfcs/pull/1584)
- Rust Issue: [39412](https://github.com/rust-lang/rust/issues/39412)

# Summary
[summary]: #summary

Declarative macros 2.0. A replacement for `macro_rules!`. This is mostly a
placeholder RFC since many of the issues affecting the new macro system are
(or will be) addressed in other RFCs. This RFC may be expanded at a later date.

Currently in this RFC:

* That we should have a new declarative macro system,
* a new keyword for declaring macros (`macro`).

In other RFCs:

* Naming and modularisation (#1561).

To come in separate RFCs:

* more detailed syntax proposal,
* hygiene improvements,
* more ...

Note this RFC does not involve procedural macros (aka syntax extensions).


# Motivation
[motivation]: #motivation

There are several changes to the declarative macro system which are desirable but
not backwards compatible (See [RFC 1561](https://github.com/rust-lang/rfcs/pull/1561)
for some changes to macro naming and modularisation, I would also like to
propose improvements to hygiene in macros, and some improved syntax).

In order to maintain Rust's backwards compatibility guarantees, we cannot change
the existing system (`macro_rules!`) to accommodate these changes. I therefore
propose a new declarative macro system to live alongside `macro_rules!`.

Example (possible) improvements:

```rust
// Naming (RFC 1561)

fn main() {
    a::foo!(...);
}

mod a {
    // Macro privacy (TBA)
    pub macro foo { ... }
}
```

```rust
// Relative paths (part of hygiene reform, TBA)

mod a {
    pub macro foo { ... bar() ... }
    fn bar() { ... }
}

fn main() {
    a::foo!(...);  // Expansion calls a::bar
}
```

```rust
// Syntax (TBA)

macro foo($a: ident) => {
    return $a + 1;
}
```

I believe it is extremely important that moving to the new macro system is as
straightforward as possible for both macro users and authors. This must be the
case so that users make the transition to the new system and we are not left
with two systems forever.

A goal of this design is that for macro users, there is no difference in using
the two systems other than how macros are named. For macro authors, most macros
that work in the old system should work in the new system with minimal changes.
Macros which will need some adjustment are those that exploit holes in the
current hygiene system.


# Detailed design
[design]: #detailed-design

There will be a new system of declarative macros using similar syntax and
semantics to the current `macro_rules!` system.

A declarative macro is declared using the `macro` keyword. For example, where a
macro `foo` is declared today as `macro_rules! foo { ... }`, it will be declared
using `macro foo { ... }`. I leave the syntax of the macro body for later
specification.

## Nomenclature

Throughout this RFC, I use 'declarative macro' to refer to a macro declared
using declarative (and domain specific) syntax (such as the current
`macro_rules!` syntax). The 'declarative macros' name is in opposition to
'procedural macros', which are declared as Rust programs. The specific
declarative syntax using pattern matching and templating is often referred to as
'macros by example'.

'Pattern macro' has been suggested as an alternative for 'declarative macro'.

# Drawbacks
[drawbacks]: #drawbacks

There is a risk that `macro_rules!` is good enough for most users and there is
low adoption of the new system. Possibly worse would be that there is high
adoption but little migration from the old system, leading to us having to
support two systems forever.


# Alternatives
[alternatives]: #alternatives

Make backwards incompatible changes to `macro_rules!`. This is probably a
non-starter due to our stability guarantees. We might be able to make something
work if this was considered desirable.

Limit ourselves to backwards compatible changes to `macro_rules!`. I don't think
this is worthwhile. It's not clear we can make meaningful improvements without
breaking backwards compatibility.

Use `macro!` instead of `macro` (proposed in an earlier version of this RFC).

Don't use a keyword - either make `macro` not a keyword or use a different word
for declarative macros.

Live with the existing system.


# Unresolved questions
[unresolved]: #unresolved-questions

What to do with `macro_rules`? We will need to maintain it at least until `macro`
is stable. Hopefully, we can then deprecate it (some time will be required to
migrate users to the new system). Eventually, I hope we can remove `macro_rules!`.
That will take a long time, and would require a 2.0 version of Rust to strictly
adhere to our stability guarantees.
