- Feature Name: prepublish
- Start Date: 2017-03-22
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC proposes the concept of *prepublication dependencies* for Cargo. These
dependencies augment a crate index (like crates.io) with new versions of crates
that have not yet been published to the index. Dependency resolution then works
*as if* those prepublished versions actually existed in the
index. Prepublication dependencies thus act as a kind of "staging index".

Prepublication makes it possible to perform integration testing within a large
crate graph before publishing anything to crates.io, and without requiring
dependencies to be switched from the crates.io index to git branches. It can, to
a degree, simulate an "atomic" change across a large number of crates and
repositories, which can then actually be landed in a piecemeal, non-atomic
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

It's likely the a couple of Cargo's existing features have already come to mind
as potential solutions to the challenges above. But the existing features suffer
from a Goldilocks problem:

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
support a new section for prepublication:

```toml
[prepublish]
xml-rs = { path = "path/to/fork" }
```

The listed dependencies must be path or git dependencies (though see
[Unresolved Questions][unresolved] for the multi-index case). Cargo will load
the crates and extract version information, supplementing the ambient index with
the version it finds. If the same version *already* exists in the ambient index,
the prepublication will act just like `[replace]`, replacing its source with the
one specified in the `[prepublish]` section. However, unlike `[replace]`,
Cargo will issue a warning in this case, since this situation is an indication
that the prepublication is ready to be removed.

Like `[replace]`, the `[prepublish]` section is only taken into account for the
root crate; allowing it to accumulate anywhere in the crate dependency graph
creates intractable problems for dependency resolution. Cargo will also refuse
to publish crates containing a `[prepublish]` section to crates.io

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

### Scenario: prepublishing a new minor version

Now, suppose that `foo` needs some changes to `xml-rs`, but we want to check
that all of Servo compiles before pushing the changes through.

First, we change `Cargo.toml` for `foo`:

```toml
[prepublish]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.9.2" }

[dependencies]
xml-rs = "0.9.2"
```

For `servo`, we also need to record the prepublication, but don't need to modify
or introduce any `xml-rs` dependencies; it's enough to be using the fork of
`foo`, which we would be anyway:

```toml
[prepublish]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.9.2" }

[dependencies]
foo = { git = "https://github.com/aturon/foo", branch = "fix-xml" }
```

With this setup:

- When compiling `foo`, Cargo will resolve the `xml-rs` dependency to `0.9.2`, and
retrieve the source from the specified git branch.

- When compiling `servo`, Cargo will again resolve *two* versions of `xml-rs`,
this time `0.9.2` and `0.8.0`, and for the former it will use the source from
the git branch.

The Cargo.toml files that needed to be changed here span from the crate that
actually cares about the new version (`foo`) upward to the root of the crate we
want to do integration testing for (`servo`); no sibling crates needed to be
changed.

Once `xml-rs` version `0.9.2` is actually published, we can remove the
`[prepublish]` sections, and Cargo will warn us that this needs to be done.

### Scenario: prepublishing a new major version

What happens if `foo` instead needs to make a breaking change to `xml-rs`? The
workflow is identical. For `foo`:

```toml
[prepublish]
xml-rs = { git = "https://github.com/aturon/xml-rs", branch = "0.10.0" }

[dependencies]
xml-rs = "0.10.0"
```

For `servo`:

```toml
[prepublish]
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

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Prepublication is a feature intended for large-scale projects spanning many
repos and crates, where you want to make something like an atomic change across
the repos. As such, it should likely be explained in a dedicated section for
large-scale Cargo usage, which would also include build system integration and
other related topics.

The mechanism itself is straightforward enough that a handful of examples (as in
this RFC) is generally enough to explain it. In the docs, these examples should
be spelled out in greater detail.

# Drawbacks
[drawbacks]: #drawbacks

This feature adds yet another knob around where, exactly, Cargo is getting its
source and version information. In particular, its similarity to `[replace]`
means the two features are likely to be confused. One saving grace is that
`[replace]` emphatically does not allow version numbers to be changed; it's very
tailored to surgical patches.

Fortunately, because both features are rarely used, are only used for very large
projects, and cannot be published to crates.io, the knobs are largely invisible
to the vast majority of Cargo users, who are unaffected by them.

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

- It does not provide an easy-to-understand picture of what the crates will look
  like after the relevant dependencies are published. In particular, you can't
  use the usual resolution algorithm to understand what's going on with version
  resolution.

- It does not provide any warning when the replaced crate has actually been
  published to the index, which could lead to silent divergences depending on
  which root crate you're compiling.

# Unresolved questions
[unresolved]: #unresolved-questions

There are two unresolved questions, both about possible future extensions.

First: it would be extremely helpful to provide a first-class workflow for
forking a dependency and making the necessary changes to Cargo.toml for
prepublication, and for fixing things up when publication actually occurs. That
shouldn't be hard to do, but is out of scope for this RFC.

Second: we may eventually want to use multiple crate indexes within a Cargo.toml
file, and we'll need some way to express *which* we're talking about with
prepublication. However, this will also be the case for standard dependencies,
so this RFC assumes that any solution will cover both cases.
