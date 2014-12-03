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

This RFC also covers the topic of altering the meaning of the stability
attributes which Rust uses today to signal the stability of the standard
library and how this interacts with feature staging. Finally, the
interaction between feature staging, experimental features, and Cargo
will be discussed.

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

## Stability Attributes

Inspired by the node.js stability levels, Rust has five levels of
stability attributes, `#[stable]`, `#[unstable]`, `#[experimental]`,
`#[deprecated]`, and unmarked. Over time, however, two primary drawbacks
have arisen:

1. The distinction between `#[unstable]` and `#[experimental]` is murky
   at best and it's unclear when an API transitions from one to the other.
2. As discussed below, the standard library will not expose
   experimental/unstable functionality on the stable and beta release
   channels. Users of the nightly channel, however, have no way to
   select *which* unstable functionality they would like from the
   standard library. A warn-by-default lint will simply warn the crate
   on any and all usage of experimental functionality. This coarse
   granularity can lead to opting into more features than one would like
   when testing out the nightly builds.

To handle these two problems, as well as play into some of the designs
listed below, this RFC proposes the following modifications to today's
stability attributes:

* Remove `#[experimental]` and recommend all users use `#[unstable]`
  instead.
* Alter the syntax of all attributes to `#[level(description = "...")]`
  to allow more metadata inside each `level`. A deprecation warning will
  be provided for migration, but an unused attribute warning will be
  issued eventually.
* Add `feature = "name"` to the `#[unstable]` attribute metadata. This
  signals that the attribute applies to a feature named `name`.
* Add `since = "version"` to the `#[stable]` attribute metadata. This
  signals the version of the language since which the API was stable.

With these modifications, new API surface area becomes a new "language
feature" which is controlled via the `#[feature]` attribute just like
other normal language features. The compiler will disallow all usage of
`#[experimental(feature = "foo")]` apis unless the current crate
declares `#![feature(foo)]`. This enables crates to declare what API
features of the standard library they rely on without opting in to all
experimental API features.

### API Lifecycle

These attributes alter the process of how new APIs are added to the
standard library slightly. First an API will be proposed via the RFC
process, and a name for the API feature being added will be assigned at
that time. When the RFC is accepted, the API will be added to the
standard library with an `#[experimental(feature = "...")]`attribute
indicating what feature the API was assigned to.

After receiving test coverage from nightly users (who have opted into
the feature) or thorough review, the API will be moved from
`#[experimental]` to `#[stable(since = "...")]` where the version listed
is the next stable version of the compiler that the API will be included
in. Note that this is two releases ahead of the current stable compiler
due to the beta period. For example, if the current stable channel is
1.1.0, then new accepted APIs will be tagged with `#[stable(since =
"1.3.0")]` because the current beta channel is the `1.2.0` release and
the nightly channel is the `1.3.0` release.

### Checking `#[feature]`

The names of features will no longer be a hardcoded list in the compiler
due to the free-form nature of the `#[experimental]` feature names.
Instead, the compiler will perform the following steps when inspecting
`#[feature]` attributes lists:

1. The compiler will discover all `#![feature]` directives
   enabled for the crate and calculate a list of all enabled features.
2. While compiling, all experimental language features used will be
   removed from this list. If a used feature is note enabled, then an
   error is generated.
3. A new pass, the stability pass, will be extracted from the current
   stability lint pass to detect usage of all experimental APIs. If an
   experimental API is used, an error is generated if the feature is not
   used, and otherwise the feature is removed from the list.
4. If the remaining list of enabled features is not empty, then the
   features were not used when compiling the current crate. The compiler
   will generate an error in this case unconditionally.

These steps ensure that the `#[feature]` attribute is used exhaustively
and will check experimental API and language features.

## Feature staging

In builds of Rust distributed through the 'beta' and 'stable' release
channels, it is impossible to turn on experimental features
by writing the `#[feature(...)]` attribute. This is accomplished
primarily through a new lint called `experimental_features`.
This lint is set to `allow` by default in nightlies and `forbid` in beta
and stable releases.

The `experimental_features` lint simply looks for all 'feature'
attributes and emits the message 'experimental feature'.

The decision to set the feature staging lint is driven by a new field
of the compilation `Session`, `disable_staged_features`. When set to
true the lint pass will configure the feature staging lint to
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
unstable APIs for crates.

With staged features disabled, the Rust build itself is not possible,
and some portion of the test suite will fail. To build the compiler
itself and keep the test suite working the build system activates
a hack via environment variables to disable the feature staging lint,
a mechanism that is not be available under typical use. The build
system additionally includes a way to run the test suite with the
feature staging lint enabled, providing a means of tracking what
portion of the test suite can be run without invoking experimental
features.

The prelude causes complications with this scheme because prelude
injection presently uses two feature gates: globs, to import the
prelude, and phase, to import the standard `macro_rules!` macros. In
the short term this will be worked-around with hacks in the
compiler. It's likely that these hacks can be removed before 1.0 if
globs and `macro_rules!` imports become stable.

## Features and Cargo

Over time, it has become clear that with an ever-growing number of Rust
releases that crates will want to be able to manage what versions of
rust they indicate they can be compiled with. Some specific use cases are:

* Although upgrades are highly encouraged, not all users upgrade
  immediately. Cargo should be able to help out with the process of
  downloading a new dependency and indicating that a newer version of
  the Rust compiler is required.
* Not all users will be able to continuously upgrade. Some enterprises,
  for example, may upgrade rarely for technical reasons. In doing so,
  however, a large portion of the crates.io ecosystem becomes unusable
  once accepted features begin to propagate.
* Developers may wish to prepare new releases of libraries during the
  beta channel cycle in order to have libraries ready for the next
  stable release. In this window, however, published versions will not
  be compatible with the current stable compiler (they use new
  features).

To solve this problem, Cargo and crates.io will grow the knowledge of
the minimum required Rust language version required to compile a crate.
Currently the Rust language version coincides with the version of the
`rustc` compiler.

To calculate this information, Cargo will compile crates just before
publishing. In this process, the Rust compiler will record all used
language features as well as all used `#[stable]` APIs. Each compiler
will contain archival knowledge of what stable version of the compiler
language features were added to, and each `#[stable]` API has the
`since` metadata to tell which version of the compiler it was released
in. The compiler will calculate the maximum of all these versions
(language plus library features) to pass to Cargo. If any `#[feature]`
directive is detected, however, the required Rust language version is
"nightly".

Cargo will then pass this required language version to crates.io which
will both store it in the index as well as present it as part of the UI.
Each crate will have a "badge" indicating what version of the Rust
compiler is needed to compile it. The "badge" may indicate that the
nightly or beta channels must be used if the version required has not
yet been released (this happens when a crate is published on a
non-stable channel). If the required language version is "nightly", then
the crate will permanently indicate that it requires the "nightly"
version of the language.

When resolving dependencies, Cargo will discard all incompatible
candidates based on the version of the available compiler. This will
enable authors to publish crates which rely on the current beta channel
while not interfering with users taking advantage of the stable channel.

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

With respect to stability attributes and Cargo, the proposed design is
very specific to the standard library and the Rust compiler without
being intended for use by third-party libraries. It is planned to extend
Cargo's own support for features (distinct from Rust features) to enable
this form of feature development in a first-class method through Cargo.
At this time, however, there are no concrete plans for this design and
it is unlikely to happen soon.

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
