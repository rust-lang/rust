- Feature Name: Allow `lifetime` specifiers to be passed to macros
- Start Date: 2016-04-22
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add a `lifetime` specifier for `macro_rules!` patterns, that matches any valid
lifetime.

# Motivation
[motivation]: #motivation

Certain classes of macros are completely impossible without the ability to pass
lifetimes. Specifically, anything that wants to implement a trait from inside of
a macro is going to need to deal with lifetimes eventually. They're also
commonly needed for any macros that need to deal with types in a more granular
way than just `ty`.

Since a lifetime is a single token, there is currently no way to accept one
without an explicit matcher. Something like `'$lifetime:ident` will fail to
compile.

# Detailed design
[design]: #detailed-design

This RFC proposes adding `lifetime` as an additional specifier to
`macro_rules!` (alternatively: `life` or `lt`). Since a lifetime acts very much
like an identifier, and can appear in almost as many places, it can be handled
almost identically. A preliminary implementation can be found at
https://github.com/rust-lang/rust/pull/33135

# Drawbacks
[drawbacks]: #drawbacks

None

# Alternatives
[alternatives]: #alternatives

A more general specifier, such as a "type parameter list", which would roughly
map to `ast::Generics` would cover most of the cases that matching lifetimes
individually would cover.

# Unresolved questions
[unresolved]: #unresolved-questions

None
