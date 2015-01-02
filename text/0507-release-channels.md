- Start Date: 2014-10-27
- RFC PR: [rust-lang/rfcs#507](https://github.com/rust-lang/rfcs/pull/507)
- Rust Issue: [rust-lang/rust#20445](https://github.com/rust-lang/rust/issues/20445)

# Summary

This RFC describes changes to the Rust release process, primarily the
division of Rust's time-based releases into 'release channels',
following the 'release train' model used by e.g. Firefox and Chrome;
as well as 'feature staging', which enables the continued development
of unstable language features and libraries APIs while providing
strong stability guarantees in stable releases.

It also redesigns and simplifies stability attributes to better
integrate with release channels and the other stability-moderating
system in the language, 'feature gates'. While this version of
stability attributes is only suitable for use by the standard
distribution, we leave open the possibility of adding a redesigned
system for the greater cargo ecosystem to annotate feature stability.

Finally, it discusses how Cargo may leverage feature gates to
determine compatibility of Rust crates with specific revisions of the
Rust language.

# Motivation

We soon intend to [provide stable releases][1] of Rust that offer
backwards compatibility with previous stable releases. Still, we
expect to continue developing new features at a rapid pace for some
time to come. We need to be able to provide these features to users
for testing as they are developed while also proving strong stability
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

* It provides a testing ground for unstable features in the
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

Development cycles are reduced to six weeks from the current twelve.

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

During the beta cycles, as with the normal release cycles, primary
development will be on the nightly branch, with only bugfixes on the
beta branch.

## Feature staging

In builds of Rust distributed through the 'beta' and 'stable' release
channels, it is impossible to turn on unstable features
by writing the `#[feature(...)]` attribute. This is accomplished
primarily through a new lint called `unstable_features`.
This lint is set to `allow` by default in nightlies and `forbid` in beta
and stable releases (and by the `forbid` setting cannot be disabled).

The `unstable_features` lint simply looks for all 'feature'
attributes and emits the message 'unstable feature'.

The decision to set the feature staging lint is driven by a new field
of the compilation `Session`, `disable_staged_features`. When set to
true the lint pass will configure the feature staging lint to
'forbid', with a `LintSource` of `ReleaseChannel`. When a
`ReleaseChannel` lint is triggered, in addition to the lint's error
message, it is accompanied by the note 'this feature may not be used
in the {channel} release channel', where `{channel}` is the name of
the release channel.

In feature-staged builds of Rust, rustdoc sets
`disable_staged_features` to *`false`*. Without doing so, it would not
be possible for rustdoc to successfully run against e.g. the
accompanying std crate, as rustdoc runs the lint pass. Additionally,
in feature-staged builds, rustdoc does not generate documentation for
unstable APIs for crates (read below for the impact of feature staging
on unstable APIs).

With staged features disabled, the Rust build itself is not possible,
and some portion of the test suite will fail. To build the compiler
itself and keep the test suite working the build system activates
a hack via environment variables to disable the feature staging lint,
a mechanism that is not be available under typical use. The build
system additionally includes a way to run the test suite with the
feature staging lint enabled, providing a means of tracking what
portion of the test suite can be run without invoking unstable
features.

The prelude causes complications with this scheme because prelude
injection presently uses two feature gates: globs, to import the
prelude, and phase, to import the standard `macro_rules!` macros. In
the short term this will be worked-around with hacks in the
compiler. It's likely that these hacks can be removed before 1.0 if
globs and `macro_rules!` imports become stable.

## Merging stability attributes and feature gates

In addition to the feature gates that, in conjuction with the
aforementioned `unstable_features` lint, manage the stable evolution
of *language* features, Rust *additionally* has another independent
system for managing the evolution of *library* features, 'stability
attributes'. This system, inspired by node.js, divides APIs into a
number of stability levels: `#[experimental]`, `#[unstable]`,
`#[stable]`, `#[frozen]`, `#[locked]`, and `#[deprecated]`, along with
unmarked functions (which are in most cases considered unstable).

As a simplifying measure stability attributes are unified with feature
gates, and thus tied to release channels and Rust language versions.

* All existing stability attributes are removed of any semantic
  meaning by the compiler. Existing code that uses these attributes
  will continue to compile, but neither rustc nor rustdoc will
  interpret them in any way.
* New `#[staged_unstable(...)]`, `#[staged_stable(...)]`,
  and `#[staged_deprecated(...)]` attributes are added.
* All three require a `feature` parameter,
  e.g. `#[staged_unstable(feature = "chicken_dinner")]`. This signals
  that the item tagged by the attribute is part of the named feature.
* The `staged_stable` and `staged_deprecated` attributes require an
  additional parameter `since`, whose value is equal to a *version of
  the language* (where currently the language version is equal to the
  compiler version), e.g. `#[stable(feature = "chicken_dinner", since
  = "1.6")]`.

All stability attributes continue to support an optional `description`
parameter.

The intent of adding the 'staged_' prefix to the stability attributes
is to leave the more desirable attribute names open for future use.

With these modifications, new API surface area becomes a new "language
feature" which is controlled via the `#[feature]` attribute just like
other normal language features. The compiler will disallow all usage
of `#[staged_unstable(feature = "foo")]` APIs unless the current crate
declares `#![feature(foo)]`. This enables crates to declare what API
features of the standard library they rely on without opting in to all
unstable API features.

Examples of APIs tagged with stability attributes:

```
#[staged_unstable(feature = "a")]
fn foo() { }

#[staged_stable(feature = "b", since = "1.6")]
fn bar() { }

#[staged_stable(feature = "c", since = "1.6")]
#[staged_deprecated(feature = "c", since = "1.7")]
fn baz() { }
```

Since *all* feature additions to Rust are associated with a language
version, source code can be finely analyzed for language
compatibility. Association with distinct feature names leads to a
straightforward process for tracking the progression of new features
into the language. More detail on these matters below.

Some additional restrictions are enforced by the compiler as a sanity
check that they are being used correctly.

* The `staged_deprecated` attribute *must* be paired with a
  `staged_stable` attribute, enforcing that the progression of all
  features is from 'staged_unstable' to 'staged_stable' to
  'staged_deprecated' and that the version in which the feature was
  promoted to stable is recorded and maintained as well as the version
  in which a feature was deprecated.
* Within a crate, the compiler enforces that for all APIs with the
  same feature name where any are marked `staged_stable`, all are
  either `staged_stable` or `staged_deprecated`. In other words, no
  single feature may be partially promoted from `unstable` to
  `stable`, but features may be partially deprecated. This ensures
  that no APIs are accidentally excluded from stabilization and that
  entire features may be considered either 'unstable' or 'stable'.

It's important to note that these stability attributes are *only known
to be useful to the standard distribution*, because of the explicit
linkage to language versions and release channels. There is though no
mechanism to explicitly forbid their use outside of the standard
distribution. A general mechanism for indicating API stability
will be reconsidered in the future.

### API lifecycle

These attributes alter the process of how new APIs are added to the
standard library slightly. First an API will be proposed via the RFC
process, and a name for the API feature being added will be assigned
at that time. When the RFC is accepted, the API will be added to the
standard library with an `#[staged_unstable(feature =
"...")]`attribute indicating what feature the API was assigned to.

After receiving test coverage from nightly users (who have opted into
the feature) or thorough review, all APIs with a given feature will be
changed from `staged_unstable` to `staged_stable`, adding `since =
"..."` to mark the version in which the promotion occurred, and the
feature is considered stable and may be used on the stable release
channel.

When a stable API becomes deprecated the `staged_deprecated` attribute
is added in addition to the existing `staged_stable` attribute, as
well recording the version in which the deprecation was performed with
the `since` parameter.

(Occassionally unstable APIs may be deprecated for the sake of easing
user transitions, in which case they receive both the `staged_stable`
and `staged_deprecated` attributes at once.)

### Checking `#[feature]`

The names of features will no longer be a hardcoded list in the compiler
due to the free-form nature of the `#[staged_unstable]` feature names.
Instead, the compiler will perform the following steps when inspecting
`#[feature]` attributes lists:

1. The compiler will discover all `#![feature]` directives
   enabled for the crate and calculate a list of all enabled features.
2. While compiling, all unstable language features used will be
   removed from this list. If a used feature is not enabled, then an
   error is generated.
3. A new pass, the stability pass, will be extracted from the current
   stability lint pass to detect usage of all unstable APIs. If an
   unstable API is used, an error is generated if the feature is not
   used, and otherwise the feature is removed from the list.
4. If the remaining list of enabled features is not empty, then the
   features were not used when compiling the current crate. The compiler
   will generate an error in this case unconditionally.

These steps ensure that the `#[feature]` attribute is used exhaustively
and will check unstable language and library features.

## Features, Cargo and version detection

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
the minimum required Rust language version required to compile a
crate. Currently the Rust language version coincides with the version
of the `rustc` compiler.

In the absense of user-supplied information about minimum language
version requirements, *Cargo will attempt to use feature information
to determine version compatibility*: by knowing in which version each
feature of the language and each feature of the library was
stabilized, and by detecting every feature used by a crate, rustc can
determine the minimum version required; and rustc may assume that the
crate will be compatible with future stable releases. There are two
caveats: first, conditional compilation makes it not possible in some
cases to detect all features in use, which may result in Cargo
detecting a minumum version less than that required on all
platforms. For this and other reasons Cargo will allow the minimum
version to be specified manually. Second, rustc can not make any
assumptions about compatibility across major revisions of the
language.

To calculate this information, Cargo will compile crates just before
publishing. In this process, the Rust compiler will record all used
language features as well as all used `#[staged_stable]` APIs. Each
compiler will contain archival knowledge of what stable version of the
compiler language features were added to, and each `#[staged_stable]`
API has the `since` metadata to tell which version of the compiler it
was released in. The compiler will calculate the maximum of all these
versions (language plus library features) to pass to Cargo. If any
`#[feature]` directive is detected, however, the required Rust
language version is "nightly".

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
[Cargo] feature that enables Rust code generation. `macro_rules!`
*is* expected to be stable by 1.0 though.

[Cargo]: https://github.com/rust-lang/rfcs/pull/403
[1]: http://blog.rust-lang.org/2014/10/30/Stability.html

With respect to stability attributes and Cargo, the proposed design is
very specific to the standard library and the Rust compiler without
being intended for use by third-party libraries. It is planned to extend
Cargo's own support for features (distinct from Rust features) to enable
this form of feature development in a first-class method through Cargo.
At this time, however, there are no concrete plans for this design and
it is unlikely to happen soon.

The attribute syntax for declaring feature names is different for
declaring feature names (a string) and for turning them on (an ident).
This is done as a judgement call that in each context the given syntax
looks best, and accepting that since this is a feature that is not
intended for general use the discrepancy is not a major problem.

Having Cargo do version detection through feature analysis is known
not to be foolproof, and may present further unknown obstacles.

# Alternatives

Leave feature gates and unstable APIs exposed to the stable
channel, as precedented by Haskell, web vendor prefixes, and node.js.

Make the beta channel a compromise between the nightly and stable
channels, allowing some set of unstable features and APIs. This
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

If stability attributes are only for std, that takes away the
`#[deprecated]` attribute from Cargo libs, which is more clearly
applicable.

What mechanism ensures that all API's have stability coverage? Probably
the will just default to unstable with some 'default' feature name.

# See Also

* [Stability as a deliverable][1]
* [Prior work week discussion][2]
* [Prior detailed description of process changes][3]

[1]: http://blog.rust-lang.org/2014/10/30/Stability.html
[2]: https://github.com/rust-lang/meeting-minutes/blob/master/workweek-2014-08-18/versioning.md)
[3]: http://discuss.rust-lang.org/t/rfc-impending-changes-to-the-release-process/508
