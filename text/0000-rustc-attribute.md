- Start Date: 2015-01-11
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Feature gate attributes of the type `#[rustc_*]` and `#[rustc]` to allow for backwards compatibility of builtin attributes.

# Motivation

Interpreting the current backwards compatibility rules strictly, it's not possible to add any further
language features that use new attributes. For example, if we wish to add a feature that expands
the attribute `#[awesome_deriving(Encodable)]` into an implementation of `Encodable`, any existing code that
contains uses of the `#[awesome_deriving]` attribute might be broken. While such attributes are useless in release 1.0 code
(since syntax extensions aren't allowed yet), we still have a case of code that stops compiling after an update of a release build.


# Detailed design

We deny the usage of any attributes with a name of `rustc` or a name starting with `rustc_` (unless a feature gate, `rustc_attributes` is enabled).

Whenever we define a new attribute that can be used outside of rustc (`#[must_use]` or `#[repr(..)]` are examples of preexisting attributes that do this),
we can give it a name starting with `rustc_`, eg `#[rustc_attr_name]` (an alternative is `#[rustc(attr_name)]`, it's best to reserve both for now).
We then whitelist the attribute from being denied; so that one will not need to enable the feature gate to use it in external crates.


This is fairly simple to achieve.

 - Add a feature gate for mentioning attributes that match the given patterns. Disallow its usage in release builds.
 - Have a builtin check for such attributes that will error when they are used, _unless_ they are in a whitelist or if the gate is opened
 - The whitelist will contain a list of language-exported custom attributes that are allowed to be used in release code.
    We can add more complicated per-attribute feature-gate checking as needed (eg for `#[rustc_on_unimplemented]`,
    which is builtin and exported but behind a feature gate).

Note that this RfC does not impose any rules on future attributes defined by the compiler (except that they must be backwards compatible).
One is free to use `#[rustc_attr_name]`, `#[rustc(attr_name)]`, or perhaps something like `#[rustc::attr_name]` if in the future we get
arbitrary token trees in attributes (or at least some form of namespacing).

# Drawbacks

I don't see much of a drawback (except that the alternatives below might be more lucrative)

# Alternatives

## Forbid `unused_attributes`

This is an alternative that is quite feasible. We simply make unused attributes a hard (unsilencable, perhaps feature gate silencing) error for release,
and the problem is solved. Compiler-defined attributes can be whitelisted for `unused_attributes` if necessary, but a random
attribute floating around in 1.0-release code will not be allowed (there's no reason to have it anyway, except perhaps for static analysis code)

For this, we may have to move the unused attribute check to somewhere post-lints; currently other lints may run after it and an attribute that is actually used
isn't marked as such.

This might be more work to implement than the main RfC (and is more drastic); but I don't see any major issues with this a priori except for the aforementioned
hurdle for static analysis. If the community is okay with it I'd love to make this the main proposal.

## Do nothing

If we do nothing we can still manage to add new attributes, however we will need to invent new syntax for it. This will probably be in the form of basic namespacing support
(`#[rustc::awesome_deriving]`) or arbitrary token tree support (the use case will probably still end up looking something like `#[rustc::awesome_deriving]`)


# Unresolved questions

Which proposal to use — disallowing `#[rustc_*]` and `#[rustc]` attributes, or just `#[forbid(unused_attribute)]`ing everything.

The main proposal can be tweaked to just disallow the `#[rustc()]` attribute (or just the `rustc_` prefix).

The names of the prefix and the feature gate could peraps be improved.