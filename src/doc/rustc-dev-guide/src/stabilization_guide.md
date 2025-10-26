# Request for stabilization

**NOTE**: This page is about stabilizing *language* features. For stabilizing *library* features, see [Stabilizing a library feature].

[Stabilizing a library feature]: ./stability.md#stabilizing-a-library-feature

Once an unstable feature has been well-tested with no outstanding concerns, anyone may push for its stabilization, though involving the people who have worked on it is prudent. Follow these steps:

## Write an RFC, if needed

If the feature was part of a [lang experiment], the lang team generally will want to first accept an RFC before stabilization.

[lang experiment]: https://lang-team.rust-lang.org/how_to/experiment.html

## Documentation PRs

<a id="updating-documentation"></a>

The feature might be documented in the [`Unstable Book`], located at [`src/doc/unstable-book`]. Remove the page for the feature gate if it exists. Integrate any useful parts of that documentation in other places.

Places that may need updated documentation include:

- [The Reference]: This must be updated, in full detail, and a member of the lang-docs team must review and approve the PR before the stabilization can be merged.
- [The Book]: This is updated as needed. If you're not sure, please open an issue on this repository and it can be discussed.
- Standard library documentation: This is updated as needed. Language features often don't need this, but if it's a feature that changes how idiomatic examples are written, such as when `?` was added to the language, updating these in the library documentation is important. Review also the keyword documentation and ABI documentation in the standard library, as these sometimes needs updates for language changes.
- [Rust by Example]: This is updated as needed.

Prepare PRs to update documentation involving this new feature for the repositories mentioned above. Maintainers of these repositories will keep these PRs open until the whole stabilization process has completed. Meanwhile, we can proceed to the next step.

## Write a stabilization report

Author a stabilization report using the [template found in this repository][srt].

The stabilization reports summarizes:

- The main design decisions and deviations since the RFC was accepted, including both decisions that were FCP'd or otherwise accepted by the language team as well as those being presented to the lang team for the first time.
    - Often, the final stabilized language feature has significant design deviations from the original RFC. That's OK, but these deviations must be highlighted and explained carefully.
- The work that has been done since the RFC was accepted, acknowledging the main contributors that helped drive the language feature forward.

The [*Stabilization Template*][srt] includes a series of questions that aim to surface connections between this feature and lang's subteams (e.g. types, opsem, lang-docs, etc.) and to identify items that are commonly overlooked.

[srt]: ./stabilization_report_template.md

The stabilization report is typically posted as the main comment on the stabilization PR (see the next section).

## Stabilization PR

Every feature is different, and some may require steps beyond what this guide discusses.

Before the stabilization will be considered by the lang team, there must be a complete PR to the Reference describing the feature, and before the stabilization PR will be merged, this PR must have been reviewed and approved by the lang-docs team.

### Updating the feature-gate listing

There is a central listing of unstable feature-gates in [`compiler/rustc_feature/src/unstable.rs`]. Search for the `declare_features!`  macro. There should be an entry for the feature you are aiming to stabilize, something like (this example is taken from [rust-lang/rust#32409]:

```rust,ignore
// pub(restricted) visibilities (RFC 1422)
(unstable, pub_restricted, "CURRENT_RUSTC_VERSION", Some(32409)),
```

The above line should be moved to [`compiler/rustc_feature/src/accepted.rs`]. Entries in the `declare_features!` call are sorted, so find the correct place. When it is done, it should look like:

```rust,ignore
// pub(restricted) visibilities (RFC 1422)
(accepted, pub_restricted, "CURRENT_RUSTC_VERSION", Some(32409)),
// note that we changed this
```

(Even though you will encounter version numbers in the file of past changes, you should not put the rustc version you expect your stabilization to happen in, but instead use `CURRENT_RUSTC_VERSION`.)

### Removing existing uses of the feature-gate

Next, search for the feature string (in this case, `pub_restricted`) in the codebase to find where it appears. Change uses of `#![feature(XXX)]` from the `std` and any rustc crates (this includes test folders under `library/` and `compiler/` but not the toplevel `tests/` one) to be `#![cfg_attr(bootstrap, feature(XXX))]`. This includes the feature-gate only for stage0, which is built using the current beta (this is needed because the feature is still unstable in the current beta).

Also, remove those strings from any tests (e.g. under `tests/`). If there are tests specifically targeting the feature-gate (i.e., testing that the feature-gate is required to use the feature, but nothing else), simply remove the test.

### Do not require the feature-gate to use the feature

Most importantly, remove the code which flags an error if the feature-gate is not present (since the feature is now considered stable). If the feature can be detected because it employs some new syntax, then a common place for that code to be is in `compiler/rustc_ast_passes/src/feature_gate.rs`. For example, you might see code like this:

```rust,ignore
gate_all!(pub_restricted, "`pub(restricted)` syntax is experimental");
```

The `gate_all!` macro reports an error if the `pub_restricted` feature is not enabled. It is not needed now that `pub(restricted)` is stable.

For more subtle features, you may find code like this:

```rust,ignore
if self.tcx.features().async_fn_in_dyn_trait() { /* XXX */ }
```

This `pub_restricted` field (named after the feature) would ordinarily be false if the feature flag is not present and true if it is. So transform the code to assume that the field is true. In this case, that would mean removing the `if` and leaving just the `/* XXX */`.

```rust,ignore
if self.tcx.sess.features.borrow().pub_restricted { /* XXX */ }
becomes
/* XXX */

if self.tcx.sess.features.borrow().pub_restricted && something { /* XXX */ }
 becomes
if something { /* XXX */ }
```

[rust-lang/rust#32409]: https://github.com/rust-lang/rust/issues/32409
[std-guide-stabilization]: https://std-dev-guide.rust-lang.org/feature-lifecycle/stabilization.html
[src-version]: https://github.com/rust-lang/rust/blob/master/src/version
[forge-versions]: https://forge.rust-lang.org/#current-release-versions
[forge-release-process]: https://forge.rust-lang.org/release/process.html
[`compiler/rustc_feature`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_feature/index.html
[`compiler/rustc_feature/src/accepted.rs`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_feature/src/accepted.rs
[`compiler/rustc_feature/src/unstable.rs`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_feature/src/unstable.rs
[The Reference]: https://github.com/rust-lang/reference
[The Book]: https://github.com/rust-lang/book
[Rust by Example]: https://github.com/rust-lang/rust-by-example
[`Unstable Book`]: https://doc.rust-lang.org/unstable-book/index.html
[`src/doc/unstable-book`]: https://github.com/rust-lang/rust/tree/master/src/doc/unstable-book

## Team nominations

When opening the stabilization PR, CC the lang team and its advisors (`@rust-lang/lang @rust-lang/lang-advisors`) and any other teams to whom the feature is relevant, e.g.:

- `@rust-lang/types`, for type system interactions.
- `@rust-lang/opsem`, for interactions with unsafe code.
- `@rust-lang/compiler`, for implementation robustness.
- `@rust-lang/libs-api`, for changes to the standard library API or its guarantees.
- `@rust-lang/lang-docs`, for questions about how this should be documented in the Reference.

After the stabilization PR is opened with the stabilization report, wait a bit for any immediate comments. When such comments "simmer down" and you feel the PR is ready for consideration by the lang team, [nominate the PR](https://lang-team.rust-lang.org/how_to/nominate.html) to get it on the agenda for consideration in an upcoming lang meeting.

If you are not a `rust-lang` organization member, you can ask your assigned reviewer to CC the relevant teams on your behalf.

## Propose FCP on the PR

After the lang team and other relevant teams review the stabilization, and after you have answered any questions they may have had, a member of one of the teams may propose to accept the stabilization by commenting:

```text
@rfcbot fcp merge
```

Once enough team members have reviewed, the PR will move into a "final comment period" (FCP). If no new concerns are raised, this period will complete and the PR can be merged after implementation review in the usual way.

## Reviewing and merging stabilizations

On a stabilization, before giving it the `r+`, ensure that the PR:

- Matches what the team proposed for stabilization and what is documented in the Reference PR.
- Includes any changes the team decided to request along the way in order to resolve or avoid concerns.
- Is otherwise exactly what is described in the stabilization report and in any relevant RFCs or prior lang FCPs.
- Does not expose on stable behaviors other than those specified, accepted for stabilization, and documented in the Reference.
- Has sufficient tests to convincingly demonstrate these things.
- Is accompanied by a PR to the Reference than has been reviewed and approved by a member of lang-docs.

In particular, when reviewing the PR, keep an eye out for any user-visible details that the lang team failed to consider and specify. If you find one, describe it and nominate the PR for the lang team.
