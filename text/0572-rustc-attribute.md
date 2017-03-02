- Start Date: 2015-01-11
- RFC PR: [#572](https://github.com/rust-lang/rfcs/pull/572)
- Rust Issue: [#22203](https://github.com/rust-lang/rust/issues/22203)

# Summary

Feature gate unused attributes for backwards compatibility.

# Motivation

Interpreting the current backwards compatibility rules strictly, it's not possible to add any further
language features that use new attributes. For example, if we wish to add a feature that expands
the attribute `#[awesome_deriving(Encodable)]` into an implementation of `Encodable`, any existing code that
contains uses of the `#[awesome_deriving]` attribute might be broken. While such attributes are useless in release 1.0 code
(since syntax extensions aren't allowed yet), we still have a case of code that stops compiling after an update of a release build.


# Detailed design

We add a feature gate, `custom_attribute`, that disallows the use of any attributes not defined by the compiler or consumed in any other way.

This is achieved by elevating the `unused_attribute` lint to a feature gate check (with the gate open, it reverts to being a lint). We'd also need to ensure that it runs after all the other lints (currently it runs as part of the main lint check and might warn about attributes which are actually consumed by other lints later on).

Eventually, we can try for a namespacing system as described below, however with unused attributes feature gated, we need not worry about it until we start considering stabilizing plugins.

# Drawbacks

I don't see much of a drawback (except that the alternatives below might be more lucrative). This might make it harder for people who wish to use custom attributes for static analysis in 1.0 code.

# Alternatives

## Forbid `#[rustc_*]` and `#[rustc(...)]` attributes

(This was the original proposal in the RfC)

This is less restrictive for the user, but it restricts us to a form of namespacing for any future attributes which we may wish to introduce. This is suboptimal, since by the time plugins stabilize (which is when user-defined attributes become useful for release code) we may add many more attributes to the compiler and they will all have cumbersome names.

## Do nothing

If we do nothing we can still manage to add new attributes, however we will need to invent new syntax for it. This will probably be in the form of basic namespacing support
(`#[rustc::awesome_deriving]`) or arbitrary token tree support (the use case will probably still end up looking something like `#[rustc::awesome_deriving]`)

This has the drawback that the attribute parsing and representation will need to be overhauled before being able to add any new attributes to the compiler.

# Unresolved questions

Which proposal to use — disallowing `#[rustc_*]` and `#[rustc]` attributes, or just `#[forbid(unused_attribute)]`ing everything.

The name of the feature gate could peraps be improved.