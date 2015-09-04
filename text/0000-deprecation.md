- Feature Name: Public Stability
- Start Date: 2015-09-03
- RFC PR: 
- Rust Issue: 

# Summary

This RFC proposes to make the stability attributes `#[deprecate]`, `#[stable]`
and `#[unstable]` publicly available, removing some and adding other 
restrictions while keeping everything mostly the same for APIs shipped with 
Rust.

# Motivation

Library authors want a way to evolve their APIs without too much breakage. To 
this end, Rust has long employed the aforementioned attributes. Now that Rust
is somewhat stable, it's time to open them up so that others can use them.

A pre-RFC on rust-users has seen a good number of supportive voices, which
suggests that the feature will improve the life of rust library authors
considerably.

# Detailed design

Add another stability level variant `Undefined`, to be used whenever a
`#[deprecate]` attribute is without a `#[stable]` or `#[unstable]`. This lifts
the restriction to have the latter attributes whenever the former is used. To
keep the restriction for ASWRs, we add an `undefined_stability` lint that is 
`Allow` by default, but set to `Warn` in the Rust build process, that catches 
`Undefined` stability attributes (can be done within the Stability lint pass).

Remove the rust API restriction on `#[deprecate]`, `#[stable]` and 
`#[unstable]`.

On all attributes the `version` field of the attribute should be checked to be 
valid semver as per [RFC 
#1122](https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md). 
It is per this RFC redefined to mean the version of the crate (or rust 
distribution, for `std`) as declared in `Cargo.toml`.

The `issue` field of the `#[`(`un`)`stable]` attributes is defined per this RFC
to mean the suffix to a URL to the issue tracker (it may also be the complete
URL). Optionally rustdoc may link the issue from the documentation; a new 
`--issue-tracker-url-prefix=...` option will be prefixed to all links.

The `feature` field is defined to contain a feature name to be used with 
`cfg(feature = "...")`. To check this, Cargo could put the list of *available*
features in a space-separated `CRATE_FEATURES_AVAILABLE` environment variable.
Alternative build processes can also set this. To simplify things, putting a
`"*"` in the environment variable should disable the check. Otherwise the
stability lint can check if the feature has been declared.

The `reason` field is defined to contain a human-readable text with a 
suggestion what to use instead and a rationale. This is how the field is used
currently. See the Alternatives section for a less conservative (but more 
work-intensive) proposal.

The language reference should be extended to describe this feature.

# Drawbacks

* Work to be done will take time not to invest in other improvements
* There could be attribute definitions in the codebase that do not adhere to
the outlined design, and would have to be changed to fit. It is unclear whether
this is a real drawback
* Once the feature is public, we can no longer change its design
* Someone could misuse the API to e.g. add malicious links into their rustdoc.
However this is possible via plain links even now

# Alternatives

* Do nothing
* Optionally the deprecation lint chould check the current version as set by
cargo in the CARGO_CRATE_VERSION environment variable (the rust build process 
should set this environment variable, too). This would allow future 
deprecations to be shown in the docs early, but not warned against by the
stability lint (there could however be a `future-deprecation` lint that should
be `Allow` by default).
* The `reason` field definition could be reduced to stating the *rationale* for 
deprecating the API item. A new `instead` field then contains the full path to 
a replacement item (trait, method, function, etc.). Since this path is well 
defined, it can be checked against. However, some provision needs to be made to 
allow those paths to be extended with a crate (e.g. for items that have been 
moved to different crates). The upside is that this would open up the 
possibility for rustdoc to link to the replacement, the downside is that the
check could potentially be costly.

# Unresolved questions

* Is the current design (as outlined herein) good enough to be made public? 
* What other restrictions should we introduce now to avoid being bound to a 
possibly flawed design?
