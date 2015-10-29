- Feature Name: Public Stability
- Start Date: 2015-09-03
- RFC PR: 
- Rust Issue: 

# Summary

This RFC proposes to allow library authors to use a `#[deprecated]` attribute,
with optional `since = "`*version*`"`, `reason = "`*free text*`"` and 
`use = "`*substitute declaration*`"` fields. The compiler can then
warn on deprecated items, while `rustdoc` can document their deprecation
accordingly.

# Motivation

Library authors want a way to evolve their APIs; which also involves 
deprecating items. To do this cleanly, they need to document their intentions 
and give their users enough time to react.

Currently there is no support from the language for this oft-wanted feature
(despite a similar feature existing for the sole purpose of evolving the Rust
standard library). This RFC aims to rectify that, while giving a pleasant
interface to use while maximizing usefulness of the metadata introduced.

# Detailed design

Public API items (both plain `fn`s, methods, trait- and inherent 
`impl`ementations as well as `const` definitions, type definitions, struct
fields and enum variants) can be given a `#[deprecated]` attribute. All
possible fields are optional:

* `since` is defined to contain the version of the crate at the time of
deprecating the item, following the semver scheme. It makes no sense to put a 
version number higher than the current newest version here, and this is not 
checked (but could be by external lints, e.g. 
[rust-clippy](https://github.com/Manishearth/rust-clippy).
* `reason` should contain a human-readable string outlining the reason for
deprecating the item. While this field is not required, library authors are
strongly advised to make use of it to convey the reason for the deprecation to 
users of their library. The string is interpreted as plain unformatted text 
(for now) so that rustdoc can include it in the item's documentation without 
messing up the formatting.
* `use`, if included, must be the import path (or a comma-separated list of 
paths) to a set of API items that will replace the functionality of the 
deprecated item. All crates in scope can be reached by this path. E.g. let's 
say my `foo()` item was superceded by either the `bar()` or `baz()` functions
in the `bar` crate, I can `#[deprecate(use="bar::{bar,baz}")] foo()`, as long 
as I have the `bar` crate in the library path. Rustc checks if the item is 
actually available, otherwise returning an error.

On use of a *deprecated* item, `rustc` will `warn` of the deprecation. Note 
that during Cargo builds, warnings on dependencies get silenced. Note that 
while this has the upside of keeping things tidy, it has a downside when it 
comes to deprecation:

Let's say I have my `llogiq` crate that depends on `foobar` which uses a
deprecated item of `serde`. I will never get the warning about this unless I
try to build `foobar` directly. We may want to create a service like `crater`
to warn on use of deprecated items in library crates, however this is outside
the scope of this RFC.

`rustdoc` will show deprecation on items, with a `[deprecated]`
box that may optionally show the version, reason and/or link to the replacement 
if available.

The language reference will be extended to describe this feature as outlined
in this RFC. Authors shall be advised to leave their users enough time to react
before *removing* a deprecated item.

The internally used feature can either be subsumed by this or possibly renamed
to avoid a name clash.

# Drawbacks

* The required checks for the `since` and `use` fields are potentially
quite complex.
* Once the feature is public, we can no longer change its design

# Alternatives

* Do nothing
* make the `since` field required and check that it's a single version
* Optionally the deprecation lint could check the current version as set by
cargo in the CARGO_CRATE_VERSION environment variable (the rust build process 
should set this environment variable, too). This would allow future 
deprecations to be shown in the docs early, but not warned against by the
stability lint (there could however be a `future-deprecation` lint that should
be `Allow` by default)
* require either `reason` or `use` be present
* `reason` could include markdown formatting
* The `use` could simply be plain text, which would remove much of the
complexity here
* The `use` field could be left out and added later. However, this would
lead people to describe a replacement in the `reason` field, as is already
happening in the case of rustc-private deprecation
* Optionally, `cargo` could offer a new dependency category: "doc-dependencies"
which are used to pull in other crates' documentations to link them (this is
obviously not only relevant to deprecation).

# Unresolved questions

* What other restrictions should we introduce now to avoid being bound to a 
possibly flawed design?
* How should the multiple values in the `use` field work? Just split by
comma or some other delimiter?
* Can / Should the `std` library make use of the `#[deprecated]` extensions?
* Bikeshedding: Are the names good enough?
