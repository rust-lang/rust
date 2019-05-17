# Implement New Feature

When you want to implement a new significant feature in the compiler,
you need to go through this process to make sure everything goes
smoothly.

## The @rfcbot (p)FCP process

When the change is small and uncontroversial, then it can be done
with just writing a PR and getting r+ from someone who knows that
part of the code. However, if the change is potentially controversial,
it would be a bad idea to push it without consensus from the rest
of the team (both in the "distributed system" sense to make sure
you don't break anything you don't know about, and in the social
sense to avoid PR fights).

If such a change seems to be too small to require a full formal RFC
process (e.g. a big refactoring of the code, or a
"technically-breaking" change, or a "big bugfix" that basically
amounts to a small feature) but is still too controversial or
big to get by with a single r+, you can start a pFCP (or, if you
don't have r+ rights, ask someone who has them to start one - and
unless they have a concern themselves, they should).

Again, the pFCP process is only needed if you need consensus - if you
don't think anyone would have a problem with your change, it's ok to
get by with only an r+. For example, it is OK to add or modify
unstable command-line flags or attributes without an pFCP for
compiler development or standard library use, as long as you don't
expect them to be in wide use in the nightly ecosystem.

You don't need to have the implementation fully ready for r+ to ask
for a pFCP, but it is generally a good idea to have at least a proof
of concept so that people can see what you are talking about.

That starts a "proposed final comment period" (pFCP), which requires
all members of the team to sign off the FCP. After they all do so,
there's a week long "final comment period" where everybody can comment,
and if no new concerns are raised, the PR/issue gets FCP approval.

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
gated - they can't be used without a enabling a feature gate
(`#[feature(foo)]`), which can't be done in a stable/beta compiler.
See the [stability in code] section for the technical details.

Eventually, after we gain enough experience using the feature,
make the necessary changes, and are satisfied, we expose it to
the world using the stabilization process described [here].
Until then, the feature is not set in stone: every part of the
feature can be changed, or the feature might be completely
rewritten or removed. Features are not supposed to gain tenure
by being unstable and unchanged for a year.

<a name = "tracking-issue"></a>
###  Tracking Issues

To keep track of the status of an unstable feature, the
experience we get while using it on nightly, and of the
concerns that block its stabilization, every feature-gate
needs a tracking issue.

General discussions about the feature should be done on
the tracking issue.

For features that have an RFC, you should use the RFC's
tracking issue for the feature.

For other features, you'll have to make a tracking issue
for that feature. The issue title should be "Tracking issue
for YOUR FEATURE".

For tracking issues for features (as opposed to future-compat
warnings), I don't think the description has to contain
anything specific. Generally we put the list of items required
for stabilization using a github list, e.g.

```txt
    **Steps:**

    - [ ] Implement the RFC (cc @rust-lang/compiler -- can anyone write
          up mentoring instructions?)
    - [ ] Adjust documentation ([see instructions on forge][doc-guide])
    - Note: no stabilization step here.
```

<a name="stability-in-code"></a>
##  Stability in code

The below steps needs to be followed in order to implement
a new unstable feature:

1. Open a [tracking issue] -
   if you have an RFC, you can use the tracking issue for the RFC.

2. Pick a name for the feature gate (for RFCs, use the name
   in the RFC).

3. Add a feature gate declaration to `libsyntax/feature_gate.rs`
   in the active `declare_features` block:

```rust,ignore
    // description of feature
    (active, $feature_name, "$current_nightly_version", Some($tracking_issue_number), $edition)
```

where `$edition` has the type `Option<Edition>`, and is typically
just `None`.

For example:

```rust,ignore
    // allow '|' at beginning of match arms (RFC 1925)
(   active, match_beginning_vert, "1.21.0", Some(44101), None),
```

The current version is not actually important – the important
version is when you are stabilizing a feature.

4. Prevent usage of the new feature unless the feature gate is set.
   You can check it in most places in the compiler using the
   expression `tcx.features().$feature_name` (or
   `sess.features_untracked().$feature_name` if the
   tcx is unavailable)

    If the feature gate is not set, you should either maintain
    the pre-feature behavior or raise an error, depending on
    what makes sense.

5. Add a test to ensure the feature cannot be used without
   a feature gate, by creating `feature-gate-$feature_name.rs`
   and `feature-gate-$feature_name.stderr` files under the
   `src/test/ui/feature-gates` directory.

6. Add a section to the unstable book, in
   `src/doc/unstable-book/src/language-features/$feature_name.md`.

7. Write a lots of tests for the new feature.
   PRs without tests will not be accepted!

8. Get your PR reviewed and land it. You have now successfully
   implemented a feature in Rust!

[value the stability of Rust]: https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md
[stability in code]: #stability-in-code
[here]: https://rust-lang.github.io/rustc-guide/stabilization_guide.html
[tracking issue]: #tracking-issue
