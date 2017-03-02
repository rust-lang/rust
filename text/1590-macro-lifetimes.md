- Feature Name: Allow `lifetime` specifiers to be passed to macros
- Start Date: 2016-04-22
- RFC PR: https://github.com/rust-lang/rfcs/pull/1590
- Rust Issue: https://github.com/rust-lang/rust/issues/34303

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

Since a lifetime is a single token, the only way to match against a lifetime is
by capturing it as `tt`. Something like `'$lifetime:ident` would fail to
compile. This is extremely limiting, as it becomes difficult to sanitize input,
and `tt` is extremely difficult to use in a sequence without using awkward
separators.

# Detailed design
[design]: #detailed-design

This RFC proposes adding `lifetime` as an additional specifier to
`macro_rules!` (alternatively: `life` or `lt`). As it is a single token, it is
able to be followed by any other specifier. Since a lifetime acts very much
like an identifier, and can appear in almost as many places, it can be handled
almost identically.

A preliminary implementation can be found at
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
