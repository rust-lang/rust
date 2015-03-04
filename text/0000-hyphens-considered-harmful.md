- Feature Name: `hyphens_considered_harmful`
- Start Date: 2015-03-05
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Disallow hyphens in package and crate names. Propose a clear transition path for existing packages.

# Motivation

Currently, Cargo packages and Rust crates both allow hyphens in their names. This is not good, for two reasons:

1.  **Usability**: Since hyphens are not allowed in identifiers, anyone who uses such a crate must rename it on import:

    ```rust
    extern crate "rustc-serialize" as rustc_serialize;
    ```

    This boilerplate confers no additional meaning, and is a common source of confusion for beginners.

2.  **Consistency**: Nowhere else do we allow hyphens in names, so having them in crates is inconsistent with the rest of the language.

For these reasons, we should work to remove this feature before the beta.

However, as of January 2015 there are 589 packages with hyphens on crates.io. It is unlikely that simply removing hyphens from the syntax will work, given all the code that depends on them. In particular, we need a plan that:

* Is easy to implement and understand;

* Accounts for the existing packages on crates.io; and

* Gives as much time as possible for users to fix their code.

# Detailed design

1. On **crates.io**:

    + Reject all further uploads for hyphenated names. Packages with hyphenated *dependencies* will still be allowed though.

    + On the server, migrate all existing hyphenated packages to underscored names. Keep the old packages around for compatibility, but hide them from search. To keep things simple, only the `name` field will change; dependencies will stay as they are.

2. In **Cargo**:

    + Continue allowing hyphens in package names, but treat them as having underscores internally. Warn the user when this happens.

      This applies to both the package itself and its dependencies. For example, imagine we have an `apple-fritter` package that depends on `rustc-serialize`. When Cargo builds this package, it will instead fetch `rustc_serialize` and build `apple_fritter`.

3. In **rustc**:

    + As with Cargo, continue allowing hyphens in `extern crate`, but rewrite them to underscores in the parser. Warn the user when this happens.

    + Do *not* allow hyphens in other contexts, such as the `#[crate_name]` attribute or `--crate-name` and `--extern` options.

      > Rationale: These options are usually provided by external tools, which would break in strange ways if rustc chooses a different name.

4. Announce the change on the users forum and /r/rust. Tell users to update to the latest Cargo and rustc, and to begin transitioning their packages to the new system. Party.

5. Some time between the beta and 1.0 release, remove support for hyphens from Cargo and rustc.

## C dependency (`*-sys`) packages

[RFC 403] introduced a `*-sys` convention for wrappers around C libraries. Under this proposal, we will use `*_sys` instead.

[RFC 403]: https://github.com/rust-lang/rfcs/blob/master/text/0403-cargo-build-command.md

# Drawbacks

## Code churn

While most code should not break from these changes, there will be much churn as maintainers fix their packages. However, the work should not amount to more than a simple find/replace. Also, because old packages are migrated automatically, maintainers can delay fixing their code until they need to publish a new version.

## Loss of hyphens

There are two advantages to keeping hyphens around:

* Aesthetics: Hyphens do look nicer than underscores.

* Namespacing: Hyphens are often used for pseudo-namespaces. For example in Python, the Django web framework has a wealth of addon packages, all prefixed with `django-`.

The author believes the disadvantages of hyphens outweigh these benefits.

# Alternatives

## Do nothing

As with any proposal, we can choose to do nothing. But given the reasons outlined above, the author believes it is important that we address the problem before the beta release.

## Disallow hyphens in crates, but allow them in packages

What we often call "crate name" is actually two separate concepts: the *package name* as seen by Cargo and crates.io, and the *crate name* used by rustc and `extern crate`. While the two names are usually equal, Cargo lets us set them separately.

For example, if we have a package named `lily-valley`, we can rename the inner crate to `lily_valley` as follows:

```toml
[package]
name = "lily-valley"  # Package name
# ...

[lib]
name = "lily_valley"  # Crate name
```

This will let us import the crate as `extern crate lily_valley` while keeping the hyphenated name in Cargo.

But while this solution solves the usability problem, it still leaves the package and crate names inconsistent. Given the few use cases for hyphens, it is unclear whether this solution is better than just disallowing them altogether.

## Make `extern crate` match fuzzily

Alternatively, we can have the compiler consider hyphens and underscores as equal while looking up a crate. In other words, the crate `flim-flam` would match both `extern crate flim_flam` and `extern crate "flim-flam" as flim_flam`. This will let us keep the hyphenated names, without having to rename them on import.

The drawback to this solution is complexity. We will need to add this special case to the compiler, guard against conflicting packages on crates.io, and explain this behavior to newcomers. That's too much work to support a marginal use case.

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

However, on prototyping this proposal, the author found it too complex and fraught with edge cases. Banning hyphens outright would be much easier to implement and understand.

# Unresolved questions

None so far.
