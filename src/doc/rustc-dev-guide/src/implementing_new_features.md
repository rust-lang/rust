# Implementing new language features

<!-- toc -->

When you want to implement a new significant feature in the compiler,
you need to go through this process to make sure everything goes
smoothly.

**NOTE: this section is for *language* features, not *library* features,
which use [a different process].**

See also [the Rust Language Design Team's procedures][lang-propose] for
proposing changes to the language.

[a different process]: ./stability.md
[lang-propose]: https://lang-team.rust-lang.org/how_to/propose.html

## The @rfcbot FCP process

When the change is small and uncontroversial, then it can be done
with just writing a PR and getting an r+ from someone who knows that
part of the code. However, if the change is potentially controversial,
it would be a bad idea to push it without consensus from the rest
of the team (both in the "distributed system" sense to make sure
you don't break anything you don't know about, and in the social
sense to avoid PR fights).

If such a change seems to be too small to require a full formal RFC process
(e.g., a small standard library addition, a big refactoring of the code, a
"technically-breaking" change, or a "big bugfix" that basically amounts to a
small feature) but is still too controversial or big to get by with a single r+,
you can propose a final comment period (FCP). Or, if you're not on the relevant
team (and thus don't have @rfcbot permissions), ask someone who is to start one;
unless they have a concern themselves, they should.

Again, the FCP process is only needed if you need consensus – if you
don't think anyone would have a problem with your change, it's OK to
get by with only an r+. For example, it is OK to add or modify
unstable command-line flags or attributes without an FCP for
compiler development or standard library use, as long as you don't
expect them to be in wide use in the nightly ecosystem.
Some teams have lighter weight processes that they use in scenarios
like this; for example, the compiler team recommends
filing a Major Change Proposal ([MCP][mcp]) as a lightweight way to
garner support and feedback without requiring full consensus.

[mcp]: https://forge.rust-lang.org/compiler/proposals-and-stabilization.html#how-do-i-submit-an-mcp

You don't need to have the implementation fully ready for r+ to propose an FCP,
but it is generally a good idea to have at least a proof
of concept so that people can see what you are talking about.

When an FCP is proposed, it requires all members of the team to sign off the
FCP. After they all do so, there's a 10-day-long "final comment period" (hence
the name) where everybody can comment, and if no concerns are raised, the
PR/issue gets FCP approval.

## The logistics of writing features

There are a few "logistic" hoops you might need to go through in
order to implement a feature in a working way.

### Warning Cycles

In some cases, a feature or bugfix might break some existing programs
in some edge cases. In that case, you might want to do a crater run
to assess the impact and possibly add a future-compatibility lint,
similar to those used for
[edition-gated lints](diagnostics.md#edition-gated-lints).

### Stability

We [value the stability of Rust]. Code that works and runs on stable
should (mostly) not break. Because of that, we don't want to release
a feature to the world with only team consensus and code review -
we want to gain real-world experience on using that feature on nightly,
and we might want to change the feature based on that experience.

To allow for that, we must make sure users don't accidentally depend
on that new feature - otherwise, especially if experimentation takes
time or is delayed and the feature takes the trains to stable,
it would end up de facto stable and we'll not be able to make changes
in it without breaking people's code.

The way we do that is that we make sure all new features are feature
gated - they can't be used without enabling a feature gate
(`#[feature(foo)]`), which can't be done in a stable/beta compiler.
See the [stability in code] section for the technical details.

Eventually, after we gain enough experience using the feature,
make the necessary changes, and are satisfied, we expose it to
the world using the stabilization process described [here].
Until then, the feature is not set in stone: every part of the
feature can be changed, or the feature might be completely
rewritten or removed. Features are not supposed to gain tenure
by being unstable and unchanged for a year.

###  Tracking Issues

To keep track of the status of an unstable feature, the
experience we get while using it on nightly, and of the
concerns that block its stabilization, every feature-gate
needs a tracking issue. General discussions about the feature should be done on the tracking issue.

For features that have an RFC, you should use the RFC's
tracking issue for the feature.

For other features, you'll have to make a tracking issue
for that feature. The issue title should be "Tracking issue
for YOUR FEATURE". Use the ["Tracking Issue" issue template][template].

[template]: https://github.com/rust-lang/rust/issues/new?template=tracking_issue.md

##  Stability in code

The below steps needs to be followed in order to implement
a new unstable feature:

1. Open a [tracking issue] -
   if you have an RFC, you can use the tracking issue for the RFC.

   The tracking issue should be labeled with at least `C-tracking-issue`.
   For a language feature, a label `F-feature_name` should be added as well.

1. Pick a name for the feature gate (for RFCs, use the name
   in the RFC).

1. Add the feature name to `rustc_span/src/symbol.rs` in the `Symbols {...}` block.

   Note that this block must be in alphabetical order.

1. Add a feature gate declaration to `rustc_feature/src/unstable.rs` in the unstable
   `declare_features` block.

   ```rust ignore
   /// description of feature
   (unstable, $feature_name, "CURRENT_RUSTC_VERSION", Some($tracking_issue_number))
   ```

   If you haven't yet
   opened a tracking issue (e.g. because you want initial feedback on whether the feature is likely
   to be accepted), you can temporarily use `None` - but make sure to update it before the PR is
   merged!

   For example:

   ```rust ignore
   /// Allows defining identifiers beyond ASCII.
   (unstable, non_ascii_idents, "CURRENT_RUSTC_VERSION", Some(55467), None),
   ```

   Features can be marked as incomplete, and trigger the warn-by-default [`incomplete_features`
   lint]
   by setting their type to `incomplete`:

   [`incomplete_features` lint]: https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#incomplete-features

   ```rust ignore
   /// Allows deref patterns.
   (incomplete, deref_patterns, "CURRENT_RUSTC_VERSION", Some(87121), None),
   ```

   To avoid [semantic merge conflicts], please use `CURRENT_RUSTC_VERSION` instead of `1.70` or
   another explicit version number.

   [semantic merge conflicts]: https://bors.tech/essay/2017/02/02/pitch/

1. Prevent usage of the new feature unless the feature gate is set.
   You can check it in most places in the compiler using the
   expression `tcx.features().$feature_name()`

    If the feature gate is not set, you should either maintain
    the pre-feature behavior or raise an error, depending on
    what makes sense. Errors should generally use [`rustc_session::parse::feature_err`].
    For an example of adding an error, see [#81015].

   For features introducing new syntax, pre-expansion gating should be used instead.
   During parsing, when the new syntax is parsed, the symbol must be inserted to the
   current crate's [`GatedSpans`] via `self.sess.gated_span.gate(sym::my_feature, span)`. 
   
   After being inserted to the gated spans, the span must be checked in the 
   [`rustc_ast_passes::feature_gate::check_crate`] function, which actually denies
   features. Exactly how it is gated depends on the exact type of feature, but most 
   likely will use the `gate_all!()` macro. 

1. Add a test to ensure the feature cannot be used without
   a feature gate, by creating `tests/ui/feature-gates/feature-gate-$feature_name.rs`.
   You can generate the corresponding `.stderr` file by running `./x test 
tests/ui/feature-gates/ --bless`.

1. Add a section to the unstable book, in
   `src/doc/unstable-book/src/language-features/$feature_name.md`.

1. Write a lot of tests for the new feature, preferably in `tests/ui/$feature_name/`.
   PRs without tests will not be accepted!

1. Get your PR reviewed and land it. You have now successfully
   implemented a feature in Rust!

[`GatedSpans`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/parse/struct.GatedSpans.html
[#81015]: https://github.com/rust-lang/rust/pull/81015
[`rustc_session::parse::feature_err`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/parse/fn.feature_err.html
[`rustc_ast_passes::feature_gate::check_crate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_passes/feature_gate/fn.check_crate.html
[value the stability of Rust]: https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md
[stability in code]: #stability-in-code
[here]: ./stabilization_guide.md
[tracking issue]: #tracking-issues
[add-feature-gate]: ./feature-gates.md#adding-a-feature-gate
