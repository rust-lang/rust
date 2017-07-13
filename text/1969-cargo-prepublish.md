- Feature Name: prepublish
- Start Date: 2017-03-22
- RFC PR: [rust-lang/rfcs#1969](https://github.com/rust-lang/rfcs/pull/1969)
- Rust Issue: N/A

# Summary
[summary]: #summary

This RFC proposes the concept of *patching sources* for Cargo. Sources can be
have their existing versions of crates replaced with different copies, and
sources can also have "prepublished" crates by adding versions of a crate which
do not currently exist in the source. Dependency resolution will work *as if*
these additional or replacement crates actually existed in the original source.

One primary feature enabled by this is the ability to "prepublish" a crate to
crates.io. Prepublication makes it possible to perform integration testing
within a large crate graph before publishing anything to crates.io, and without
requiring dependencies to be switched from the crates.io index to git branches.
It can, to a degree, simulate an "atomic" change across a large number of crates
and repositories, which can then actually be landed in a piecemeal, non-atomic
fashion.

# Motivation
[motivation]: #motivation

Large Rust projects often end up pulling in dozens or hundreds of crates from
crates.io, and those crates often depend on each other as well. If the project
author wants to contribute a change to one of the crates nestled deep in the
graph (say, `xml-rs`), they face a couple of related challenges:

- Before submitting a PR upstream to `xml-rs`, they will usually want to try
  integrating the change within their project, to make sure it actually meets
  their needs and doesn't lead to unexpected problems. That might involve a
  *cascade* of changes if several crates in the graph depend on `xml-rs`. How do
  they go about this kind of integration work prior to sending a PR?

- If the change to the upstream `xml-rs` crate is breaking (would require a new
  major version), it's vital to carefully track which other crates in the graph
  have successfully been updated to this version, and which ones are still at
  the old version (and can stay there). This issue is related to the notion of
  [private dependencies](https://github.com/rust-lang/cargo/issues/2064), which
  should have a separate RFC in the near future.

- Once they're satisfied with the change to `xml-rs` (and any other intermediate
  crates), they'll need to make PRs and request a new publication to
  crates.io. But they would like to cleanly continue local development in the
  meantime, with an easy migration as each PR lands and each crate is published.

## The Goldilocks problem

It's likely that a couple of Cargo's existing features have already come to
mind as potential solutions to the challenges above. But the existing features
suffer from a Goldilocks problem:

- You might reach for git (or even path) dependencies. That would mean, for
  example, switching an `xml-rs` dependency in your crate graph from crates.io
  to point at, for example, a forked copy on github. The problem is that **this
  approach does not provide enough dependency unification**: if other parts of
  the crate graph refer to the crates.io version of `xml-rs`, it is treated as
  an entirely separate library and thus compiled separately. That in turn means
  that two crates in the graph using these distinct versions won't be able to
  talk to each other about `xml-rs` types (even when those types are identical).

- You might think that `[replace]` was designed precisely for the use case
  above. But **it provides too much dependency unification**: it reroutes *all*
  uses of a particular existing crate version to new source for the crate, even
  if there are breaking changes involved. The feature is designed for surgical
  patching of specific dependency versions.

Prepublication dependencies add another tool to this arsenal, with just the
right amount of dependency unification: the precise amount you'd get after
publication to crates.io.

# Detailed design
[design]: #detailed-design

The design itself is relatively straightforward. The Cargo.toml file will
support a new section for patching a source of crates:

```toml
[patch.crates-io]
xml-rs = { path = "path/to/fork" }
```

The listed dependencies have the same syntax as the normal `[dependencies]`
section, but they must all come form a different source than the source being
patched. For example you can't patch crates.io with other crates from
crates.io! Cargo will load the crates and extract the version information for
each dependency's name, supplementing the source specified with the version it
finds. If the same name/version pair *already* exists in the source being
patched, then this will act just like `[replace]`, replacing its source with
the one specified in the `[patch]` section.

Like `[replace]`, the `[patch]` section is only taken into account for the
root crate (or workspace root); allowing it to accumulate anywhere in the crate
dependency graph creates intractable problems for dependency resolution.

The sub-table of `[patch]` (where `crates-io` is used above) is used to
specify the source that's being patched. Cargo will know ahead of time one
identifier, literally `crates-io`, but otherwise this field will currently be
interpreted as a URL of a source. The name `crates-io` will correspond to the
crates.io index, and other urls, such as git repositories, may also be specified
for patching. Eventually it's intended we'll grow support for multiple
registries here with their own identifiers, but for now just literally
`crates-io` and other URLs are allowed.

## Examples

It's easiest to see how the feature works by looking at a few examples.

Let's imagine that `xml-rs` is currently at version `0.9.1` on crates.io, and we
have the following dependency setup:

- Crate `foo` lists dependency `xml-rs = "0.9.0"`
- Crate `bar` lists dependency `xml-rs = "0.9.1"`
- Crate `baz` lists dependency `xml-rs = "0.8.0"`
- Crate `servo` has `foo`, `bar` and `baz` as dependencies.

With this setup, the dependency graph for Servo will contain *two* versions of
`xml-rs`: `0.9.1` and `0.8.0`. That's because minor versions are coalesced;
`0.9.1` is considered a minor release against `0.9.0`, while `0.9.0` and `0.8.0`
are incompatible.

### Scenario: patching with a bugfix

Let's say that while developing `foo` we've got a lock file pointing to `xml-rs`
`0.9.0`, and we found the `0.9.0` branch of `xml-rs` that hasn't been touched
since it was published. We then find a bug in the 0.9.0 publication of `xml-rs`
which we'd like to fix.

First we'll check out `foo` locally and implement what we believe is a fix for
this bug, and next, we change `Cargo.toml` for `foo`:

```toml
[patch.crates-io]
xml-rs = { path = "../xml-rs" }
```

When compiling `foo`, Cargo will resolve the `xml-rs` dependency to `0.9.0`,
as it did before, but that version's been replaced with our local copy. The
local path dependency, which has version 0.9.0, takes precedence over the
version found in the registry.

Once we've confirmed a fix bug we then continue to run tests in `xml-rs` itself,
and then we'll send a PR to the main `xml-rs` repo. This then leads us to the
next section where a new version of `xml-rs` comes into play!

### Scenario: prepublishing a new minor version

Now, suppose that `foo` needs some changes to `xml-rs`, but we want to check
that all of Servo compiles before pushing the changes through.

First, we change `Cargo.toml` for `foo`:

```toml
[patch.crates-io]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.9.2" }

[dependencies]
xml-rs = "0.9.2"
```

For `servo`, we also need to record the prepublication, but don't need to modify
or introduce any `xml-rs` dependencies; it's enough to be using the fork of
`foo`, which we would be anyway:

```toml
[patch.crates-io]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.9.2" }
foo = { git = "https://github.com/aturon/foo", branch = "fix-xml" }
```

Note that if Servo depended directly on `foo` it would also be valid to do:

```toml
[patch.crates-io]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.9.2" }

[dependencies]
foo = { git = "https://github.com/aturon/foo", branch = "fix-xml" }
```

With this setup:

- When compiling `foo`, Cargo will resolve the `xml-rs` dependency to `0.9.2`,
  and retrieve the source from the specified git branch.

- When compiling `servo`, Cargo will again resolve *two* versions of `xml-rs`,
  this time `0.9.2` and `0.8.0`, and for the former it will use the source from
  the git branch.

The Cargo.toml files that needed to be changed here span from the crate that
actually cares about the new version (`foo`) upward to the root of the crate we
want to do integration testing for (`servo`); no sibling crates needed to be
changed.

Once `xml-rs` version `0.9.2` is actually published, we will likely be able to
remove the `[patch]` sections. This is a discrete step that must be taken by
crate authors, however (e.g. doesn't happen automatically) because the actual
published 0.9.2 may not be precisely what we thought it was going to be. For
example more changes could have been merged, it may not actually fix the bug,
etc.

### Scenario: prepublishing a breaking change

What happens if `foo` instead needs to make a breaking change to `xml-rs`? The
workflow is identical. For `foo`:

```toml
[patch.crates-io]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.10.0" }

[dependencies]
xml-rs = "0.10.0"
```

For `servo`:

```toml
[patch.crates-io]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.10.0" }

[dependencies]
foo = { git = "https://github.com/aturon/foo", branch = "fix-xml" }
```

However, when we compile, we'll now get *three* versions of `xml-rs`: `0.8.0`,
`0.9.1` (retained from the previous lockfile), and `0.10.0`. Assuming that
`xml-rs` is a public dependency used to communicate between `foo` and `bar` this
will result in a compilation error, since they are using distinct versions of
`xml-rs`. To fix that, we'll need to update `bar` to also use the new, `0.10.0`
prepublication version of `xml-rs`.

(Note that a
[private dependency](https://github.com/rust-lang/cargo/issues/2064) distinction
would help catch this issue at the Cargo level and give a maximally informative
error message).

## Impact on `Cargo.lock`

Usage of `[patch]` will perform backwards-incompatible modifications to
`Cargo.lock`, meaning that usage of `[patch]` will prevent previous versions
of Cargo from interpreting the lock file. Cargo will unconditionally resolve all
entries in the `[patch]` section to precise dependencies, encoding them all in
the lock file whether they're used or not.

Dependencies formed on crates listed in `[patch]` will then be listed directly
in Cargo.lock, and the original listed crate will not be listed. In our example
above we had:

- Crate `foo` lists dependency `xml-rs = "0.9.0"`
- Crate `bar` lists dependency `xml-rs = "0.9.1"`
- Crate `baz` lists dependency `xml-rs = "0.8.0"`

We then update the crate `foo` to have a dependency of `xml-rs = "0.10.0"`. This
causes Cargo to encode in the lock file that `foo` depends directly on the git
repository of `xml-rs` containing `0.10.0`, but it does **not** mention that
`foo` depends on the crates.io version of `xml-rs-0.10.0` (it doesn't exist!).
Note, however, that the lock file will still mention `xml-rs-0.8.0` and
`xml-rs-0.9.1` because `bar` and `baz` depend on it.

To help put some TOML where our mouth is let's say we depend on `env_logger` but
we're using `[patch]` to depend on a git version of the `log` crate, a
dependency of `env_logger`. First we'll have our `Cargo.toml` including:

```toml
# Cargo.toml
[dependencies]
env_logger = "0.4"
```

With that we'll find this in `Cargo.lock`, notably everything comes from
crates.io

```toml
# Cargo.lock
[[package]]
name = "env_logger"
version = "0.4.2"
source = "registry+https://github.com/rust-lang/crates.io-index"
dependencies = [
 "log 0.3.7 (registry+https://github.com/rust-lang/crates.io-index)",
]

[[package]]
name = "log"
version = "0.3.7"
source = "registry+https://github.com/rust-lang/crates.io-index"
```

Next up we'll add our `[patch]` section to crates.io:

```toml
# Cargo.toml
[patch.crates-io]
log = { git = 'https://github.com/rust-lang-nursery/log' }
```

and that will generate a lock file that looks (roughly) like:

```toml
# Cargo.lock
[[package]]
name = "env_logger"
version = "0.4.2"
source = "registry+https://github.com/rust-lang/crates.io-index"
dependencies = [
 "log 0.3.7 (git+https://github.com/rust-lang-nursery/log)",
]

[[package]]
name = "log"
version = "0.3.7"
source = "git+https://github.com/rust-lang-nursery/log#cb9fa28812ac27c9cadc4e7b18c221b561277289"
```

Notably `log` from crates.io *is not mentioned at all here*, and crucially so!
Additionally Cargo has the fully resolved version of the `log` patch
available to it, down to the sha of what to check out.

When Cargo rebuilds from this `Cargo.lock` it will not query the registry for
versions of `log`, instead seeing that there's an exact dependency on the git
repository (from the `Cargo.lock`) and the repository is listed as a
patch, so it'll follow that pointer.

## Impact on `[replace]`

The `[patch]` section in the manifest can in many ways be seen as a "replace
2.0". It is, in fact, strictly more expressive than the current `[replace]`
section! For example these two sections are equivalent:

```toml
[replace]
'log:0.3.7' = { git = 'https://github.com/rust-lang-nursery/log' }

# is the same as...

[patch.crates-io]
log = { git = 'https://github.com/rust-lang-nursery/log' }
```

This is not accidental! The intial development of the `[patch]` feature was
actually focused on prepublishing dependencies and was called `[prepublish]`,
but while discussing it a conclusion was reached that `[prepublish]` already
allowed replacing existing versions in a registry, but issued a warning when
doing so. It turned out that without a warning we ended up having a full-on
`[replace]` replacement!

At this time, though, it is not planned to deprecate the `[replace]` section,
nor remove it. After the `[patch]` section is implemented, if it ends up
working out this may change. If after a few cycles on stable the `[patch]`
section seems to be working well we can issue an official deprecation for
`[replace]`, printing a warning if it's still used.

Documentation, however, will immediately begin to recommend `[patch]` over
`[replace]`.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Patching is a feature intended for large-scale projects spanning many repos
and crates, where you want to make something like an atomic change across the
repos. As such, it should likely be explained in a dedicated section for
large-scale Cargo usage, which would also include build system integration and
other related topics.

The mechanism itself is straightforward enough that a handful of examples (as in
this RFC) is generally enough to explain it. In the docs, these examples should
be spelled out in greater detail.

Most notably, however, the [overriding dependenices][over] section of Cargo's
documentation will be rewritten to primarily mention `[patch]`, but
`[replace]` will be mentioned still with a recommendation to use `[patch]`
instead if possible.

[over]: http://doc.crates.io/specifying-dependencies.html#overriding-dependencies

# Drawbacks
[drawbacks]: #drawbacks

This feature adds yet another knob around where, exactly, Cargo is getting its
source and version information. In particular, it's basically deprecating
`[replace]` if it works out, and it's typically a shame to deprecate major
stable features.

Fortunately, because these features are intended to be relatively rarely used,
checked in even more rarely, are only used for very large projects, and cannot
be published to crates.io, the knobs are largely invisible to the vast majority
of Cargo users, who are unaffected by them.

# Alternatives
[alternatives]: #alternatives

The primary alternative for addressing the motivation of this RFC would be to
loosen the restrictions around `[replace]`, allowing it to arbitrarily change
the version of the crate being replaced.

As explained in the motivation section, however, such an approach does not fully
address the desired workflow, for a few reasons:

- It does not make it possible to track which crates in the dependency graph
  have successfully upgraded to a new major version of the replaced dependency,
  which could have the effect of masking important *behavioral* breaking changes
  (that still allow the crates to compile).

- It does not provide an easy-to-understand picture of what the crates will
  likely look like after the relevant dependencies are published. In particular,
  you can't use the usual resolution algorithm to understand what's going on
  with version resolution. A good example of this is the "breaking change"
  example above where we ended up with three versions of `xml-rs` after our
  prepublished version. It's crucial that 0.9.1 was still in the lock file
  because we hadn't updated that dependency on 0.9.1 yet, so it wasn't ready for
  0.10.0. With `[replace]`, however, we would only possibly be able to replace
  all usage of 0.9.1 with 0.10.0, not having an incremental solution.

# Unresolved questions
[unresolved]: #unresolved-questions

- It would be extremely helpful to provide a first-class workflow for forking a
  dependency and making the necessary changes to Cargo.toml for prepublication,
  and for fixing things up when publication actually occurs. That shouldn't be
  hard to do, but is out of scope for this RFC.
