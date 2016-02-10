- Feature Name: grammar
- Start Date: 2015-10-21
- RFC PR: [rust-lang/rfcs#1331](https://github.com/rust-lang/rfcs/pull/1331)
- Rust Issue: [rust-lang/rust#30942](https://github.com/rust-lang/rust/issues/30942)

# Summary
[summary]: #summary
[src/grammar]: https://github.com/rust-lang/rust/tree/master/src/grammar

Grammar of the Rust language should not be rustc implementation-defined. We have a formal grammar
at [src/grammar] which is to be used as the canonical and formal representation of the Rust
language.

# Motivation
[motivation]: #motivation
[#1228]: https://github.com/rust-lang/rfcs/blob/master/text/1228-placement-left-arrow.md
[#1219]: https://github.com/rust-lang/rfcs/blob/master/text/1219-use-group-as.md
[#1192]: https://github.com/rust-lang/rfcs/blob/master/text/1192-inclusive-ranges.md

In many RFCs proposing syntactic changes ([#1228], [#1219] and [#1192] being some of more recently
merged RFCs) the changes are described rather informally and are hard to both implement and
discuss which also leads to discussions containing a lot of guess-work.

Making [src/grammar] to be the canonical grammar and demanding for description of syntactic changes
to be presented in terms of changes to the formal grammar should greatly simplify both the
discussion and implementation of the RFCs. Using a formal grammar also allows us to discover and
rule out existence of various issues with the grammar changes (e.g. grammar ambiguities) during
design phase rather than implementation phase or, even worse, after the stabilisation.

# Detailed design
[design]: #detailed-design
[A-grammar]: https://github.com/rust-lang/rust/issues?utf8=✓&q=is:issue+is:open+label:A-grammar

Sadly, the [grammar][src/grammar] in question is [not quite equivalent][A-grammar] to the
implementation in rustc yet. We cannot possibly hope to catch all the quirks in the rustc parser
implementation, therefore something else needs to be done.

This RFC proposes following approach to making [src/grammar] the canonical Rust language grammar:

1. Fix the already known discrepancies between implementation and [src/grammar];
2. Make [src/grammar] a [semi-canonical grammar];
3. After a period of time transition [src/grammar] to a [fully-canonical grammar].

## Semi-canonical grammar
[semi-canonical grammar]: #semi-canonical-grammar

Once all known discrepancies between the [src/grammar] and rustc parser implementation are
resolved, [src/grammar] enters the state of being semi-canonical grammar of the Rust language.

Semi-canonical means that all new development involving syntax changes are made and discussed in
terms of changes to the [src/grammar] and [src/grammar] is in general regarded to as the canonical
grammar except when new discrepancies are discovered. These discrepancies must be swiftly resolved,
but resolution will depend on what kind of discrepancy it is:

1. For syntax changes/additions introduced after [src/grammar] gained the semi-canonical state, the
   [src/grammar] is canonical;
2. For syntax that was present before [src/grammar] gained the semi-canonical state, in most cases
   the implementation is canonical.

This process is sure to become ambiguous over time as syntax is increasingly adjusted (it is harder
to “blame” syntax changes compared to syntax additions), therefore the resolution process of
discrepancies will also depend more on a decision from the Rust team.

## Fully-canonical grammar
[fully-canonical grammar]: #fully-canonical-grammar

After some time passes, [src/grammar] will transition to the state of fully canonical grammar.
After [src/grammar] transitions into this state, for any discovered discrepancies the
rustc parser implementation must be adjusted to match the [src/grammar], unless decided otherwise
by the RFC process.

## RFC process changes for syntactic changes and additions

Once the [src/grammar] enters semi-canonical state, all RFCs must describe syntax additions and
changes in terms of the formal [src/grammar]. Discussion about these changes are also expected (but
not necessarily will) to become more formal and easier to follow.

# Drawbacks
[drawbacks]: #drawbacks

This RFC introduces a period of ambiguity during which neither implementation nor [src/grammar] are
truly canonical representation of the Rust language. This will be less of an issue over time as
discrepancies are resolved, but its an issue nevertheless.

# Alternatives
[alternatives]: #alternatives

One alternative would be to immediately make [src/grammar] a fully-canonical grammar of the Rust
language at some arbitrary point in the future.

Another alternative is to simply forget idea of having a formal grammar be the canonical grammar of
the Rust language.

# Unresolved questions
[unresolved]: #unresolved-questions

How much time should pass between [src/grammar] becoming semi-canonical and fully-canonical?
