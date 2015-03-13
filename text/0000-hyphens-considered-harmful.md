- Feature Name: `hyphens_considered_harmful`
- Start Date: 2015-03-05
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Disallow hyphens in Rust crate names, but continue allowing them in Cargo packages.

# Motivation

This RFC aims to reconcile two conflicting points of view.

First: hyphens in crate names are awkward to use, and inconsistent with the rest of the language. Anyone who uses such a crate must rename it on import:

```rust
extern crate "rustc-serialize" as rustc_serialize;
```

An earlier version of this RFC aimed to solve this issue by removing hyphens entirely.

However, there is a large amount of precedent for keeping `-` in package names. Systems as varied as GitHub, npm, RubyGems and Debian all have an established convention of using hyphens. Disallowing them would go against this precedent, causing friction with the wider community.

Fortunately, Cargo presents us with a solution. It already separates the concepts of *package name* (used by Cargo and crates.io) and *crate name* (used by rustc and `extern crate`). We can disallow hyphens in the crate name only, while still accepting them in the outer package. This solves the usability problem, while keeping with the broader convention.

# Detailed design

## Disallow hyphens in crates (only)

In **Cargo**, continue allowing hyphens in package names. But unless the `Cargo.toml` says otherwise, the inner crate name will have all hyphens replaced with underscores.

For example, if I had a package named `apple-fritter`, its crate will be named `apple_fritter` instead.

In **rustc**, enforce that all crate names are valid identifiers. With the changes in Cargo, existing hyphenated packages should continue to build unchanged.

## Identify `-` and `_` on crates.io

Right now, crates.io compares package names case-insensitively. This means, for example, you cannot upload a new package named `RUSTC-SERIALIZE` because `rustc-serialize` already exists.

Under this proposal, we will extend this logic to identify `-` and `_` as well.

## Remove the quotes from `extern crate`

Change the syntax of `extern crate` so that the crate name is no longer in quotes (e.g. `extern crate photo_finish as photo;`). This is viable now that all crate names are valid identifiers.

To ease the transition, keep the old `extern crate` syntax around, transparently mapping any hyphens to underscores. For example, `extern crate "silver-spoon" as spoon;` will be desugared to `extern crate silver_spoon as spoon;`. This syntax will be deprecated, and removed before 1.0.

# Drawbacks

## Inconsistency between packages and crates

This proposal makes package and crate names inconsistent: the former will accept hyphens while the latter will not.

However, this drawback may not be an issue in practice. As hinted in the motivation, most other platforms have different syntaxes for packages and crates/modules anyway. Since the package system is orthogonal to the language itself, there is no need for consistency between the two.

## Inconsistency between `-` and `_`

Quoth @P1start:

> ... it's also annoying to have to choose between `-` and `_` when choosing a crate name, and to remember which of `-` and `_` a particular crate uses.

I believe, like other naming issues, this problem can be addressed by conventions.

# Alternatives

## Do nothing

As with any proposal, we can choose to do nothing. But given the reasons outlined above, the author believes it is important that we address the problem before the beta release.

## Disallow hyphens in package names as well

An earlier version of this RFC proposed to disallow hyphens in packages as well. The drawbacks of this idea are covered in the motivation.

## Make `extern crate` match fuzzily

Alternatively, we can have the compiler consider hyphens and underscores as equal while looking up a crate. In other words, the crate `flim-flam` would match both `extern crate flim_flam` and `extern crate "flim-flam" as flim_flam`.

This involves much more magic than the original proposal, and it is not clear what advantages it has over it.

## Repurpose hyphens as namespace separators

Alternatively, we can treat hyphens as path separators in Rust.

For example, the crate `hoity-toity` could be imported as

```rust
extern crate hoity::toity;
```

which is desugared to:

```rust
mod hoity {
    mod toity {
        extern crate "hoity-toity" as krate;
        pub use krate::*;
    }
}
```

However, on prototyping this proposal, the author found it too complex and fraught with edge cases. For these reasons the author chose not to push this solution.

# Unresolved questions

None so far.
