- Start Date: 2014-10-27
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

This RFC describes changes to the Rust release process, primarily the
division of Rust's time-based releases into 'release channels',
following the 'release train' model used by e.g. Firefox and Chrome;
as well as 'feature staging', which enables the continued development
of experimental language features and libraries APIs while providing
strong stability guarantees in stable releases.

# Motivation

We soon intend to [provide stable releases][1] of Rust that offer
forward compatibility with future releases. Still, we expect to
continue developing new features at a rapid pace for some time to
come. We need to be able to provide these features to users for
testing as they are developed while also proving strong stability
guarantees to users.

[1]: http://blog.rust-lang.org/2014/10/30/Stability.html

# Detailed design

The Rust release process moves to a 'release train' model, in which
there are three 'release channels' through which the official Rust
binaries are published: 'nightly', 'beta', and 'stable', and these
release channels correspond to development branches.

'Nightly` is exactly as today, and where most development occurs; a
separate 'beta' branch provides time for vetting a release and fixing
bugs - particularly in backwards compatibility - before it gets wide
use. Each release cycle beta gets promoted to stable (the release),
and nightly gets promoted to beta.

The benefits of this model are a few:

* It provides a window for testing the next release before committing
  to it. Currently we release straight from the (very active) master
  branch, with almost no testing.

* It provides a window in which library developers can test their code
  against the next release, and - importantly - report unintended
  breakage of stable features.

* It provides a testing ground for experimental features in the
  nightly release channel, while allowing the primary releases to
  contain only features which are complete and backwards-compatible
  ('feature-staging').

This proposal describes the practical impact to users of the release
train, particularly with regard to feature staging. A more detailed
description of the impact on the development process is [available
elsewhere][3].

## Versioning and releases

The nature of development and releases differs between channels, as
each serves a specific purpose: nightly is for active development,
beta is for testing and bugfixing, and stable is for final releases.

Each pending version of Rust progresses in sequence through the
'nightly' and 'beta' channels before being promoted to the 'stable'
channel, at which time the final commit is tagged and that version is
considered 'released'.

Under normal circumstances, the version is only bumped on the nightly
branch, once per development cycle, with the release channel
controlling the label (`-nightly`, `-beta`) appended to the version
number. Other circumstances, such as security incidents, may require
point releases on the stable channel, the policy around which is yet
undetermined.

Builds of the 'nightly' channel are published every night based on the
content of the master branch. Each published build during a single
development cycle carries *the same version number*,
e.g. '1.0.0-nightly', though for debugging purposes rustc builds can
be uniquely identified by reporting the commit number from which they
were built. As today, published nightly artifacts are simply referred
to as 'rust-nightly' (not named after their version number). Artifacts
produced from the nightly release channel should be considered
transient, though we will maintain historical archives for convenience
of projects that occasionally need to pin to specific revisions.

Builds of the 'beta' channel are published periodically as fixes are
merged, and like the 'nightly' channel each published build during a
single development cycle retains the same version number, but can be
uniquely identified by the commit number. Beta artifacts are likewise
simply named 'rust-beta'.

We will ensure that it is convenient to perform continuous integration
of Cargo packages against the beta channel on Travis CI. This will
help detect any accidental breakage early, while not interfering with
their build status.

Stable builds are versioned and named the same as today's releases,
both with just a bare version number, e.g. '1.0.0'.  They are
published at the beginning of each development cycle and once
published are never refreshed or overwritten. Provisions for stable
point releases will be made at a future time.

## Exceptions for the 1.0.0 beta period

Under the release train model version numbers are incremented
automatically each release cycle on a predetermined schedule.  Six
weeks after 1.0.0 is released 1.1.0 will be released, and six weeks
after that 1.2.0, etc.

The release cycles approaching 1.0.0 will break with this pattern to
give us leeway to extend 1.0.0 betas for multiple cycles until we are
confident the intended stability guarantees are in place.

In detail, when the development cycle begins in which we are ready to
publish the 1.0.0 beta, we will *not* publish anything on the stable
channel, and the release on the beta channel will be called
1.0.0-beta1. If 1.0.0 betas extend for multiple cycles, the will be
called 1.0.0-beta2, -beta3, etc, before being promoted to the stable
channel as 1.0.0 and beginning the release train process in full.

## Feature staging

In builds of Rust distributed through the 'beta' and 'stable' release
channels, it is impossible to turn on experimental language features
by writing the `#[feature(...)]` attribute or to use APIs *from
libraries distributed as part of the main Rust distribution* tagged
with either `#[experimental]` or `#[unstable]`. This is accomplished
primarily through three new lints, `experimental_features`,
`staged_unstable`, and `staged_experimental`, which are set to 'allow'
by default in nightlies, and 'forbid' in beta and stable releases.

The `experimental_features` lint simply looks for all 'feature'
attributes and emits the message 'experimental feature'.

The `staged_unstable` and `staged_experimental` behave exactly like
the existing `unstable` and `experimental` lints, emitting the message
'unstable' and 'experimental', except that they only apply to crates
marked with the `#[staged_api]` attribute. If this attribute is not
present then the lints have no effect.

All crates in the Rust distribution are marked `#[staged_api]`.
Libraries in the Cargo registry are not bound to participate in
feature staging because they are not required to be
`#[staged_api]`. Crates maintained by the Rust project (the rust-lang
org on GitHub) but not included in the main Rust distribution are not
`#[staged_api]`.

The decision to set the feature staging lints is driven by a new field
of the compilation `Session`, `disable_staged_features`. When set to
true the lint pass will configure the three feature staging lints to
'forbid', with a `LintSource` of `ReleaseChannel`. Once set to
'forbid' it is not possible for code to programmaticaly disable the
lint. When a `ReleaseChannel` lint is triggered, in addition to the
lint's error message, it is accompanied by the note 'this feature may
not be used in the {channel} release channel', where `{channel}` is
the name of the release channel.

In feature-staged builds of Rust, rustdoc sets
`disable_staged_features` to *`false`*. Without doing so, it would not
be possible for rustdoc to successfully run against e.g. the
accompanying std crate, as rustdoc runs the lint pass. Additionally,
in feature-staged builds, rustdoc does not generate documentation for
experimental and unstable APIs for crates with the `#[staged_api]`
attribute.

With staged features disabled, the Rust build itself is not possible,
and some portion of the test suite will fail. To build the compiler
itself and keep the test suite working the build system activates
a hack via environment variables to disable the feature staging lints,
a mechanism that is not be available under typical use. The build
system additionally includes a way to run the test suite with the
feature staging lints enabled, providing a means of tracking what
portion of the test suite can be run without invoking experimental
features.

The prelude causes complications with this scheme because prelude
injection presently uses two feature gates: globs, to import the
prelude, and phase, to import the standard `macro_rules!` macros. In
the short term this will be worked-around with hacks in the
compiler. It's likely that these hacks can be removed before 1.0 if
globs and `macro_rules!` imports become stable.

# Drawbacks

Adding multiple release channels and reducing the release cycle from
12 to 6 weeks both increase the amount of release engineering work
required.

The major risk in feature staging is that, at the 1.0 release not
enough of the language is available to foster a meaningful library
ecosystem around the stable release. While we might expect many users
to continue using nightly releases with or without this change, if the
stable 1.0 release cannot be used in any practical sense it will be
problematic from a PR perspective. Implementing this RFC will require
careful attention to the libraries it affects.

Recognizing this risk, we must put in place processes to monitor the
compatibility of known Cargo crates with the stable release channel,
using evidence drawn from those crates to prioritize the stabilization
of features and libraries. [This work has already begun][1], with
popular feature gates being ungated, and library stabilization work
being prioritized based on the needs of Cargo crates.

Syntax extensions, lints, and any program using the compiler APIs
will not be compatible with the stable release channel at 1.0 since it
is not possible to stabilize `#[plugin_registrar]` in time. Plugins
are very popular. This pain will partially be alleviated by a proposed
[Cargo] feature that enables Rust code generation.

[Cargo]: https://github.com/rust-lang/rfcs/pull/403
[1]: http://blog.rust-lang.org/2014/10/30/Stability.html

# Alternatives

Leave feature gates and experimental APIs exposed to the stable
channel, as precedented by Haskell, web vendor prefixes, and node.js.

Make the beta channel a compromise between the nightly and stable
channels, allowing some set of experimental features and APIs. This
would allow more projects to use a 'more stable' release, but would
make beta no longer representative of the pending stable release.

# Unresolved questions

The exact method for working around the prelude's use of feature gates
is undetermined. Fixing [#18102] will complicate the situation as the
prelude relies on a bug in lint checking to work at all.

[#18102]: https://github.com/rust-lang/rust/issues/18102

Rustdoc disables the feature-staging lints so they don't cause it to
fail, but I don't know why rustdoc needs to be running lints. It may
be possible to just stop running lints in rustdoc.

# See Also

* [Stability as a deliverable][1]
* [Prior work week discussion][2]
* [Prior detailed description of process changes][3]

[1]: http://blog.rust-lang.org/2014/10/30/Stability.html
[2]: https://github.com/rust-lang/meeting-minutes/blob/master/workweek-2014-08-18/versioning.md)
[3]: http://discuss.rust-lang.org/t/rfc-impending-changes-to-the-release-process/508
