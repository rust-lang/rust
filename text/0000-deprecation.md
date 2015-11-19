- Feature Name: Public Stability
- Start Date: 2015-09-03
- RFC PR: 
- Rust Issue: 

# Summary

This RFC proposes to allow library authors to use a `#[deprecated]` attribute,
with optional `since = "`*version*`"` and `reason = "`*free text*`"`fields. The 
compiler can then warn on deprecated items, while `rustdoc` can document their 
deprecation accordingly.

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
deprecating the item, following the semver scheme. Rustc does not know about
versions, thus the content of this field is not checked (but will be by external
lints, e.g. [rust-clippy](https://github.com/Manishearth/rust-clippy).
* `reason` should contain a human-readable string outlining the reason for
deprecating the item. While this field is not required, library authors are
strongly advised to make use of it to convey the reason for the deprecation to 
users of their library. The string is interpreted as plain unformatted text 
(for now) so that rustdoc can include it in the item's documentation without 
messing up the formatting.

On use of a *deprecated* item, `rustc` will `warn` of the deprecation. Note 
that during Cargo builds, warnings on dependencies get silenced. While this has 
the upside of keeping things tidy, it has a downside when it comes to 
deprecation:

Let's say I have my `llogiq` crate that depends on `foobar` which uses a
deprecated item of `serde`. I will never get the warning about this unless I
try to build `foobar` directly. We may want to create a service like `crater`
to warn on use of deprecated items in library crates, however this is outside
the scope of this RFC.

`rustdoc` will show deprecation on items, with a `[deprecated]` box that may 
optionally show the version and reason where available.

The language reference will be extended to describe this feature as outlined
in this RFC. Authors shall be advised to leave their users enough time to react
before *removing* a deprecated item.

The internally used feature can either be subsumed by this or possibly renamed
to avoid a name clash.

# Intended Use

Crate author Anna wants to evolve her crate's API. She has found that one
type, `Foo`, has a better implementation in the `rust-foo` crate. Also she has
written a `frob(Foo)` function to replace the earlier `Foo::frobnicate(self)`
method. 

So Anna first bumps the version of her crate (because deprecation is always
done on a version change) from `0.1.1` to `0.2.1`. She also adds the following 
prefix to the `Foo` type:

```
extern crate rust_foo;

#[deprecated(since = "0.2.1", use="rust_foo::Foo", 
    reason="The rust_foo version is more advanced, and this crates' will likely be discontinued")]
struct Foo { .. }
```

Users of her crate will see the following once they `cargo update` and `build`:

```
src/foo_use.rs:27:5: 27:8 warning: Foo is marked deprecated as of version 0.2.1
src/foo_use.rs:27:5: 27:8 note: The rust_foo version is more advanced, and this crates' will likely be discontinued
```

Rust-clippy will likely gain more sophisticated checks for deprecation:

* `future_deprecation` will warn on items marked as deprecated, but with a
version lower than their crates', while `current_deprecation` will warn only on
those items marked as deprecated where the version is equal or lower to the
crates' one.
* `deprecation_syntax` will check that the `since` field really contains a
semver number and not some random string.

Clippy users can then activate the clippy checks and deactivate the standard
deprecation checks.

# Drawbacks

* Once the feature is public, we can no longer change its design

# Alternatives

* Do nothing
* make the `since` field required and check that it's a single version
* require either `reason` or `use` be present
* `reason` could include markdown formatting
* rename the `reason` field to `note` to clarify it's broader usage.
* add a `note` field and make `reason` a field with specific meaning, perhaps
even predefine a number of valid reason strings, as JEP277 currently does
* Add a `use` field containing a plain text of what to use instead
* Add a `use` field containing a path to some function, type, etc. to replace
the current feature. Currently with the rustc-private feature, people are 
describing a replacement in the `reason` field, which is clearly not the 
original intention of the field
* Optionally, `cargo` could offer a new dependency category: "doc-dependencies"
which are used to pull in other crates' documentations to link them (this is
obviously not only relevant to deprecation)

# Unresolved questions

* What other restrictions should we introduce now to avoid being bound to a 
possibly flawed design?
* Can / Should the `std` library make use of the `#[deprecated]` extensions?
* Bikeshedding: Are the names good enough?
