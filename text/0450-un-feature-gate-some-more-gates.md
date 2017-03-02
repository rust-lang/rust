- Start Date: 2014-12-02
- RFC PR: [450](https://github.com/rust-lang/rfcs/pull/450)
- Rust Issue: [19469](https://github.com/rust-lang/rust/issues/19469)

# Summary

Remove the `tuple_indexing`, `if_let`, and `while_let` feature gates and add
them to the language.

# Motivation

## Tuple Indexing

This feature has proven to be quite useful for tuples and struct variants, and
it allows for the removal of some unnecessary tuple accessing traits in the
standard library (TupleN).

The implementation has also proven to be quite solid with very few reported
internal compiler errors related to this feature.

## `if let` and `while let`

This feature has also proven to be quite useful over time. Many projects are now
leveraging these feature gates which is a testament to their usefulness.

Additionally, the implementation has also proven to be quite solid with very
few reported internal compiler errors related to this feature.

# Detailed design

* Remove the `if_let`, `while_let`, and `tuple_indexing` feature gates.
* Add these features to the language (do not require a feature gate to use them).
* Deprecate the `TupleN` traits in `std::tuple`.

# Drawbacks

Adding features to the language this late in the game is always somewhat of a
risky business. These features, while having baked for a few weeks, haven't had
much time to bake in the grand scheme of the language. These are both backwards
compatible to accept, and it could be argued that this could be done later
rather than sooner.

In general, the major drawbacks of this RFC are the scheduling risks and
"feature bloat" worries. This RFC, however, is quite easy to implement (reducing
schedule risk) and concerns two fairly minor features which are unambiguously
nice to have.

# Alternatives

* Instead of un-feature-gating before 1.0, these features could be released
  after 1.0 (if at all). The `TupleN` traits would then be required to be
  deprecated for the entire 1.0 release cycle.

# Unresolved questions

None at the moment.
