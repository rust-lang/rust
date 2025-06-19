# Request for stabilization

**NOTE**: this page is about stabilizing *language* features.
For stabilizing *library* features, see [Stabilizing a library feature].

[Stabilizing a library feature]: ./stability.md#stabilizing-a-library-feature

Once an unstable feature has been well-tested with no outstanding
concern, anyone may push for its stabilization. It involves the
following steps:

<!-- toc -->

## Documentation PRs

<a id="updating-documentation"></a>

If any documentation for this feature exists, it should be
in the [`Unstable Book`], located at [`src/doc/unstable-book`].
If it exists, the page for the feature gate should be removed.

If there was documentation there, integrating it into the
existing documentation is needed.

If there wasn't documentation there, it needs to be added.

Places that may need updated documentation:

- [The Reference]: This must be updated, in full detail.
- [The Book]: This may or may not need updating, depends.
    If you're not sure, please open an issue on this repository
    and it can be discussed.
- standard library documentation: As needed. Language features
    often don't need this, but if it's a feature that changes
    how good examples are written, such as when `?` was added
    to the language, updating examples is important.
- [Rust by Example]: As needed.

Prepare PRs to update documentation involving this new feature
for repositories mentioned above. Maintainers of these repositories
will keep these PRs open until the whole stabilization process
has completed. Meanwhile, we can proceed to the next step.

## Write a stabilization report

Author a stabilization report using the [template found in this repository][srt].

Stabilization reports summarize:

- The main design decisions and deviations since the RFC was accepted, particularly decisions that were FCP'd or otherwise accepted by the language team.
    - Quite often, the final stabilized language feature can have significant design deviations from the original RFC text.
- The work that has been done since the RFC was accepted, acknowledging the main contributors that helped drive the language feature forward.

The [*Stabilization Template*][srt] includes a series of questions that aim to surface interconnections between this feature and the various Rust teams (lang, types, etc) and also to identify items that are commonly overlooked.

[srt]: ./stabilization_report_template.md

The stabilization report is typically posted as the main comment on the stabilization PR (see the next section).

## Stabilization PR for a language feature

*This is for stabilizing language features.  If you are stabilizing a library
feature, see [the stabilization chapter of the std dev guide][std-guide-stabilization] instead.*

Here is a general guide to how to stabilize a feature --
every feature is different, of course, so some features may
require steps beyond what this guide talks about.

Note: Before we stabilize any feature, it's the rule that it
should appear in the documentation.

### Updating the feature-gate listing

There is a central listing of unstable feature-gates in
[`compiler/rustc_feature/src/unstable.rs`]. Search for the `declare_features!`
macro. There should be an entry for the feature you are aiming
to stabilize, something like (this example is taken from
[rust-lang/rust#32409]:

```rust,ignore
// pub(restricted) visibilities (RFC 1422)
(unstable, pub_restricted, "CURRENT_RUSTC_VERSION", Some(32409)),
```

The above line should be moved to [`compiler/rustc_feature/src/accepted.rs`].
Entries in the `declare_features!` call are sorted, so find the correct place.
When it is done, it should look like:

```rust,ignore
// pub(restricted) visibilities (RFC 1422)
(accepted, pub_restricted, "CURRENT_RUSTC_VERSION", Some(32409)),
// note that we changed this
```

(Even though you will encounter version numbers in the file of past changes,
you should not put the rustc version you expect your stabilization to happen in,
but instead `CURRENT_RUSTC_VERSION`)

### Removing existing uses of the feature-gate

Next search for the feature string (in this case, `pub_restricted`)
in the codebase to find where it appears. Change uses of
`#![feature(XXX)]` from the `std` and any rustc crates (this includes test folders
under `library/` and `compiler/` but not the toplevel `tests/` one) to be
`#![cfg_attr(bootstrap, feature(XXX))]`. This includes the feature-gate
only for stage0, which is built using the current beta (this is
needed because the feature is still unstable in the current beta).

Also, remove those strings from any tests (e.g. under `tests/`). If there are tests
specifically targeting the feature-gate (i.e., testing that the
feature-gate is required to use the feature, but nothing else),
simply remove the test.

### Do not require the feature-gate to use the feature

Most importantly, remove the code which flags an error if the
feature-gate is not present (since the feature is now considered
stable). If the feature can be detected because it employs some
new syntax, then a common place for that code to be is in the
same `compiler/rustc_ast_passes/src/feature_gate.rs`.
For example, you might see code like this:

```rust,ignore
gate_all!(pub_restricted, "`pub(restricted)` syntax is experimental");
```

This `gate_feature_post!` macro prints an error if the
`pub_restricted` feature is not enabled. It is not needed
now that `#[pub_restricted]` is stable.

For more subtle features, you may find code like this:

```rust,ignore
if self.tcx.features().async_fn_in_dyn_trait() { /* XXX */ }
```

This `pub_restricted` field (obviously named after the feature)
would ordinarily be false if the feature flag is not present
and true if it is. So transform the code to assume that the field
is true. In this case, that would mean removing the `if` and
leaving just the `/* XXX */`.

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

After the stabilization PR is opened with the stabilization report, wait a bit for potential immediate comments. When such immediate comments "simmer down" and you feel the PR is ready for consideration by the lang team, you can [nominate the PR](https://lang-team.rust-lang.org/how_to/nominate.html) to get it on the list for discussion in the next meeting. You should also cc the other interacting teams when applicable to review the language feature being stabilized and the stabilization report:

* `@rust-lang/types`, to look for type system interactions
* `@rust-lang/compiler`, to review implementation robustness
* `@rust-lang/opsem`, if this feature interacts with unsafe code and can create undefined behavior
* `@rust-lang/libs-api`, if there are additions to the standard library that affects standard library API or their guarantees

If you are not an organization member, you can simply ask your assigned reviewer to cc the relevant teams on your behalf.

## FCP proposed on the PR

Finally, some member of the team responsible for tracking this feature agrees with stabilizing this feature, will
start the FCP (final-comment-period) process by commenting

```text
@rfcbot fcp merge
```

The rest of the team members will review the proposal. If the final decision is to stabilize, the PR will be reviewed by the compiler team like any other PR.
