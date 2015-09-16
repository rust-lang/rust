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

* `since` is defined to contain the exact version of the crate that
deprecated the item, as defined by Cargo.toml (thus following the semver
scheme). It makes no sense to put a version number higher than the current
newest version here, and this is not checked (but could be by external
lints, e.g. [rust-clippy](https://github.com/Manishearth/rust-clippy).
To maximize usefulness, the version should be fully specified (e.g. no
wildcards or ranges).
* `reason` should contain a human-readable string outlining the reason for
deprecating the item. While this field is not required, library authors are
strongly advised to make use of it to convey the reason to users of their
library. The string is required to be plain unformatted text (for now) so that
rustdoc can include it in the item's documentation without messing up the 
formatting.
* `use` should be the full path to an API item that will replace the 
functionality of the deprecated item, optionally (if the replacement is in a 
different crate) followed by `@` and either a crate name (so that 
`https://crates.io/crates/` followed by the name is a live link) or the URL to 
a repository or other location where a surrogate can be obtained. Links must be 
plain FTP, FTPS, HTTP or HTTPS links. The intention is to allow rustdoc (and
possibly other tools in the future, e.g. IDEs) to act on the included 
information. The `use` field can have multiple values.

On use of a *deprecated* item, `rustc` should `warn` of the deprecation. Note 
that during Cargo builds, warnings on dependencies get silenced. Note that 
while this has the upside of keeping things tidy, it has a downside when it 
comes to deprecation:

Let's say I have my `llogiq` crate that depends on `foobar` which uses a
deprecated item of `serde`. I will never get the warning about this unless I
try to build `foobar` directly. We may want to create a service like `crater`
to warn on use of deprecated items in library crates, however this is outside
the scope of this RFC.

`rustdoc` should show deprecation on items, with a `[deprecated since x.y.z]`
box that may optionally show the reason and/or link to the replacement if
available.

The language reference should be extended to describe this feature as outlined
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
* Optionally the deprecation lint chould check the current version as set by
cargo in the CARGO_CRATE_VERSION environment variable (the rust build process 
should set this environment variable, too). This would allow future 
deprecations to be shown in the docs early, but not warned against by the
stability lint (there could however be a `future-deprecation` lint that should
be `Allow` by default)
* require either `reason` or `use` be present
* `reason` could include markdown formatting
* The `use` could simply be plain text, which would remove much of the
complexity here
* The `use` field contents could make use of the context in finding
replacements, e.g. extern crates, so that `time::precise_time_ns` would resolve
to the `time::precise_time_ns` API in the `time` crate, provided an
`extern crate time;` declaration is present
* The `use` field could be left out and added later. However, this would
lead people to describe a replacement in the `reason` field, as is already
happening in the case of rustc-private deprecation

# Unresolved questions

* What other restrictions should we introduce now to avoid being bound to a 
possibly flawed design?
* How should the multiple values in the `use` field work? Just split by
comma or some other delimiter?
* Can / Should the `std` library make use of the `#[deprecated]` extensions?
* Bikeshedding: Are the names good enough?
