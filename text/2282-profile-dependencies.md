- Feature Name: profile_dependencies
- Start Date: 2018-01-08
- RFC PR: [rust-lang/rfcs#2282](https://github.com/rust-lang/rfcs/pull/2282)
- Rust Issue: [rust-lang/rust#48683](https://github.com/rust-lang/rust/issues/48683)


# Summary
[summary]: #summary

Allow overriding profile keys for certain dependencies, as well as providing a way to set profiles in `.cargo/config`

# Motivation
[motivation]: #motivation

Currently the "stable" way to tweak build parameters like "debug symbols", "debug assertions", and "optimization level" is to edit Cargo.toml.

This file is typically checked in tree, so for many projects overriding things involves making
temporary changes to this, which feels hacky. On top of this, if Cargo is being called by an
encompassing build system as what happens in Firefox, these changes can seem surprising.

This also doesn't allow for much customization. For example, when trying to optimize for
compilation speed by building in debug mode, build scripts will get built in debug mode as well. In
case of complex build-time dependencies like bindgen, this can end up significantly slowing down
compilation. It would be nice to be able to say "build in debug mode, but build build dependencies
in release". Also, your program may have large dependencies that it doesn't use in critical paths,
being able to ask for just these dependencies to be run in debug mode would be nice.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation


Currently, the [Cargo guide has a section on this](http://doc.crates.io/manifest.html#the-profile-sections).

We amend this to add that you can override dependency configurations via `profile.foo.overrides`:

```toml
[profile.dev]
opt-level = 0
debug = true

# the `image` crate will be compiled with -Copt-level=3
[profile.dev.overrides.image]
opt-level = 3

# All dependencies (but not this crate itself) will be compiled
# with -Copt-level=2 . This includes build dependencies.
[profile.dev.overrides."*"]
opt-level = 2

# Build scripts and their dependencies will be compiled with -Copt-level=3
# By default, build scripts use the same rules as the rest of the profile
[profile.dev.build_override]
opt-level = 3
```

Additionally, profiles may be listed in `.cargo/config`. When building, cargo will calculate the
current profile, and if it has changed, it will do a fresh/clean build.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

In case of overlapping rules, the precedence order is that `overrides.foo`
will win over `overrides."*"` and both will win over `build_override`.

So if you specify `build_override`
it will not affect the compilation of any dependencies which are both
build-dependencies and regular dependencies. If you have

```toml
[profile.dev]
opt-level = 0
[profile.dev.build_override]
opt-level = 3
```

and the `image` crate is _both_ a build dependency and a regular dependency; it will be compiled
as per the top level `opt-level=0` rule. If you wish it to be compiled as per the build_override rule,
use a normal override rule:

```toml
[profile.dev]
opt-level = 0
[profile.dev.build_override]
opt-level = 3
[profile.dev.overrides.image]
opt-level = 3
```

This clash may not occur whilst cross compiling since two separate versions of the crate will be compiled.
(This RFC leaves the decision of whether or not to handle this up to the implementors)

It is not possible to have the same crate compiled in different modes as a build dependency and a
regular dependency within the same profile when not cross compiling. (This is a current limitation
in Cargo, but it would be nice if we could fix this)

Put succinctly, `build_override` is not able to affect anything compiled into the final binary.

`cargo build --target foo` will fail to run if `foo` clashes with the name of a profile; so avoid
giving profiles the same name as possible build targets.

When in a workspace, `"*"` will apply to all dependencies that are _not_ workspace members, you can explicitly
apply things to workspace members with `[profile.dev.overrides.membername]`.

The `panic` key cannot be specified in an override; only in the top level of a profile. Rust does not allow
the linking together of crates with different `panic` settings.

# Drawbacks
[drawbacks]: #drawbacks

This complicates cargo.

# Rationale and alternatives
[alternatives]: #alternatives

There are really two or three concerns here:

 - A stable interface for setting various profile keys (`cargo rustc -- -Clto` is not good, for example, and doesn't integrate into Cargo's target directories)
 - The ability to use a different profile for build scripts (usually, the ability to flip optimization modes; I don't think folks care as much about `-g` in build scripts)
 - The ability to use a different profile for specific dependencies

The first one can be resolved partially by stabilizing `cargo` arguments for overriding these. It
doesn't fix the target directory issue, but that might not be a major concern. Allowing profiles to
come from `.cargo/config` is another minimal solution to this for use cases like Firefox, which
wraps Cargo in another build system.

The second one can be fixed with a specific `build-scripts = release` key for profiles.

The third can't be as easily fixed, however it's not clear if that's a major need.

The nice thing about this proposal is that it is able to handle all three of these concerns. However, separate RFCs for separate features could be introduced as well.

In general there are plans for Cargo to support other build systems by making it more modular (so
that you can ask it for a build plan and then execute it yourself). Such build systems would be able to
provide the ability to override profiles themselves instead. It's unclear if the general Rust
community needs the ability to override profiles.

# Unresolved questions
[unresolved]: #unresolved-questions

- Bikeshedding the naming of the keys
- The current proposal provides a way to say "special-case all build dependencies, even if they are regular dependencies as well",
  but not "special-case all build-only dependencies" (which can be solved with a `!build_override` thing, but that's weird and unweildy)
- It would be nice to have a way for crates to _declare_ that they use a particular
  panic mode (something like `allow-panic=all` vs `allow-panic=abort`/`allow_panic=unwind`, with `all` as default)
  so that they can assume a panic mode and cargo will refuse to compile them with anything else
