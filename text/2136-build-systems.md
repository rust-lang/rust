- Feature Name: N/A
- Start Date: 2017-09-01
- RFC PR: [rust-lang/rfcs#2136](https://github.com/rust-lang/rfcs/pull/2136)
- Rust Issue: N/A

# Summary
[summary]: #summary

This **experimental RFC** lays out a high-level plan for improving Cargo's
ability to integrate with other build systems and environments. As an
experimental RFC, it opens the door to landing [unstable features] in Cargo to
try out ideas, but *not* to stabilizing those features, which will require
follow-up RFCs. It proposes a variety of features which, in total, permit a wide
spectrum of integration cases -- from customizing a single aspect of Cargo to
letting an external build system run almost the entire show.

[unstable features]: https://github.com/rust-lang/cargo/pull/4433/

# Motivation
[motivation]: #motivation

One of the first hurdles for using Rust in production is integrating it into
your organization's build system. The level of challenge depends on the level of
integration required: it's relatively painless to invoke Cargo from a makefile
and let it fully manage building Rust code, but gets harder as you want the
external build system to exert finer-grained control over how Rust code is built.
The goal of this RFC is to lay out a vision for making integration at *any* scale
much easier than it is today.

After extensive discussion with stakeholders, there appear to be two distinct
kinds of use-cases (or "customers") involved here:

- **Mixed build systems**, where building already involves a variety of
  language- or project-specific build systems. For this use case, the desire is
  to use Cargo as-is, except for some specific concerns. Those concerns take a
  variety of shapes: customizing caching, having a local crate registry, custom
  handling for native dependencies, and so on. Addressing these concerns well
  means adding new points of extensibility or control to Cargo.

- **Homogeneous build systems** like [Bazel], where there is a single prevailing
  build system and methodology that works across languages and projects and is
  expected to drive all aspects of the build. In such cases the goal of Cargo
  integration is largely *interoperability*, including easy use of the crates.io
  ecosystem and Rust-centric tooling, both of which expect Cargo-driven build
  management.

[Bazel]: https://bazel.build/

The interoperability constraints are, in actuality, hard constraints around
*any* kind of integration.

In more detail, a build system integration *must*:

- Make it easy for the outer build system to control the aspects of building
  that are under its purview (e.g. artifact management, caching, network access).
- Make it easy to depend on arbitrary crates in the crates.io ecosystem.
- Make it easy to use Rust tooling like `rustfmt` or the RLS with projects that
  depend on the external build system.

A build system integration *should*:

- Provide Cargo-based or Cargo-like workflows when developing Rust projects, so
  that documentation and guidance from the Rust community applies even when
  working within a different build system.
- To the extent possible, support Cargo concepts in a smooth, first-class way in
  the external build system (e.g. Cargo features, profiles, etc)

This RFC does not attempt to provide a detailed solution for all of the needed
extensibility points in Cargo, but rather to outline a general plan for how to
get there over time. Individual components that add significant features to
Cargo will need follow-up RFCs before stabilization.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The plan proposed in this RFC is to address the two use-cases from the
[motivation] section in parallel:

- **For the mixed build system case**, we will triage feature requests and work on
  adding further points of extensibility to Cargo based on expected impact. Each
  added point of extensibility should ease build system integration for another
  round of customers.

- **For the homoegenous build system case**, we will immediately pursue
  extensibility points that will enable the external build system to perform
  many of the tasks that Cargo does today--but while still meeting our
  interoperability constraints. We will then work on smoothing remaining rough
  edges, which have a high degree of overlap with the work on mixed build
  systems.

In the long run, these two parallel lines of work will converge, such that we
offer a complete spectrum of options (in terms of what Cargo controls versus an
external system). But they start at critically different points, and working on
those in parallel is the key to delivering value quickly and incrementally.

## A high-level model of what Cargo does

Before delving into the details of the plan, it's helpful to lay out a mental
model of the work that Cargo does today, broken into several stages:

| **Step**              | **Conceptual output**                                                                                                                                    | **Related concerns**             |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Dependency resolution | Lock file                                                                                                                                                | Custom registries, mirrors, offline/local, native deps, ... |
| Build configuration   | Cargo settings per crate in graph                                                                                                                        | Profiles                            |
| Build lowering        | A build plan: a series of steps that must be run in sequence, including rustc and binary invocations | Build scripts, plugins              |
| Build execution       | Compiled artifacts                                                                                                                                       | Caching                             |

The first stage, dependency resolution, is the most complex; it's where our
model of semver comes into play, as well as a huge list of related
concerns.

Dependency resolution produces a lockfile, which records what crates are
included in the dependency graph, coming from what sources and at what versions,
as well as interdependencies. It operates independently of the requested Cargo
workflow.

The next stage is build *configuration*, which conceptually is where things like
profiles come into play: of the crates we're going to build, we need to decide,
at a high level, "how" we're going to build them. This configuration is at the
"Cargo level of abstraction", i.e. in terms of things like profiles rather than
low-level rustc flags. There's strong desire to make this system more
expressive, for example by allowing you to always optimize certain dependencies
even when otherwise in the debug profile.

After configuration, we know at the Cargo level exactly what we want to build,
but we need to *lower* the level of abstraction into concrete, individual
steps. This is where, for example, profile information is transformed into
specific rustc flags. Lowering is done independently for each crate, and results
in a sequence of process invocations, interleaving calls to `rustc` with
e.g. running the binary for a build script. You can think of these sequences as
expanding what was previously a "compile this crate with this configuration"
node in the dependency graph into a finer-grained set of nodes for running rustc
etc.

Finally, there's the actual build *execution*, which is conceptually
straightforward: we analyze the dependency graph and existing, cached artifacts,
and then actually perform any un-cached build steps (in parallel when
possible). Of course, this is the bread-and-butter of many external build
systems, so we want to make it easy for them to tweak or entirely control this
part of the process.

The first two steps -- dependency resolution and build configuration -- need to
operate on an entire dependency graph at once. Build lowering, by contrast, can
be performed for any crate in isolation.

### Customizing Cargo

**A key point is that, in principle, each of these steps is separable from the
others**. That is, we should be able to rearchitect Cargo so that each of these
steps is managed by a distinct component, and the components have a stable --
and public! -- way of communicating with one another. That in turn will enable
replacing any particular component while keeping the others. (To be clear, the
breakdown above is just a high-level sketch; in reality, we'll need a more
nuanced and layered picture of Cargo's activities).

This RFC proposes to provide *some* means of customizing Cargo's activities at
various layers and stages. The details here are *very much* up for grabs, and
are part of the experimentation we need to do.

#### Likely design constraints

Some likely constraints for a Cargo customization/plugin system are:

- It should be possible for Rust tools (like `rustfmt`, IDEs, linters) to "call
  Cargo" to get information or artifacts in a standardized way, while remaining
  oblivious to any customizations. Ideally, `Cargo` workflows (including custom
  commands) would also work transparently.

- It should be possible to customize or swap out a *small part* of Cargo's
  behavior without understanding or reimplementing other parts.

- The interface for customization should be *forward-compatible*: existing
  plugins should continue to work with new versions of Cargo.

- It should be difficult or impossible to introduce customizations that are
  "incoherent", for example that result in unexpected differences in the way
  that `rustc` is invoked in different workflows (because, say, the testing
  workflow was customized but the normal build workflow wasn't). In other words,
  customizations are subject to *cross-cutting concerns*, which need to be
  identified and factored out.

We will iterate on the constraints to form core design principles as we
experiment.

#### A concrete example

Since the above is quite hand-wavy, it's helpful to see a very simple, concrete
example of what a customization might look like. You could imagine something
like the following for supplying manifest information from an external build
system, rather than through `Cargo.toml`:

**Cargo.toml**

```toml
[plugins.bazel]
generate-manifest = true
```

**$root/.cargo/meta.toml**

```toml
[plugins]

# These dependencies cannot themselves use plugins.
# This file is "staged" earlier than Cargo.toml

bazel = "1.0" # a regular crates.io dependency
```

**Semantics**

If any `plugins` entry in `Cargo.toml` defines a `generate-manifest` key,
whenever Cargo would be about to return the parsed results of `Cargo.toml` ,
instead:

- look for the associated plugin in `.cargo/meta.toml`, and ask it to generate the manifest
- return that instead

## Specifics for the homogeneous build system case

For homogeneous build systems, there are two kinds of code that must be dealt
with: code originally written using vanilla Cargo and a crate registry, and code
written "natively" in the context of the external build system. Any integration
has to handle the first case to have access to crates.io or a vendored mirror
thereof.

### Using crates vendored from or managed by a crate registry

Whether using a registry server or a vendored copy, if you're building Rust code
that is written using vanilla Cargo, you will at some level need to use Cargo's
dependency resolution and `Cargo.toml` files. In this case, the external build
system should invoke Cargo for *at least* the dependency resolution and build
configuration steps, and likely the build lowering step as well. In such a
world, Cargo is responsible for *planning* the build (which involves largely
Rust-specific concerns), but the external build system is responsible for
*executing* it.

A typical pattern of usage is to have a whitelist of "root dependencies" from an
external registry which will be permitted as dependencies within the
organization, often pinning to a specific version and set of Cargo
features. This whitelist can be described as a single `Cargo.toml` file, which
can then drive Cargo's dependency resolution just once for the entire registry.
The resulting lockfile can be used to guide vendoring and construction of a
build plan for consumption by the external build system.

One important concern is: how do you depend on code from other languages, which
is being managed by the external build system? That's a narrow version of a more
general question around *native dependencies*, which will be addressed
separately in a later section.

#### Workflow and interop story

On the external build system side, a rule or plugin will need to be written that
knows how to invoke Cargo to produce a build plan corresponding to a whitelisted
(and potentially vendored) registry, then translate that build plan back into
appropriate rules for the build system. Thus, when doing normal builds, the
external build system drives the entire process, but invokes Cargo for guidance
during the planning stage.

### Using crates managed by the build system

Many organization want to employ their own strategy for maintaining and
versioning code and for resolving dependencies, *in addition* to build
execution.

In this case, the big question is: how can we arrange things such that the Rust
tooling ecosystem can understand what the external build system is doing, to
gather the information needed for the tools to operate.

The possibility we'll examine here is using Cargo **purely as a conduit for
information from the external build system to Rust tools** (see Alternatives for
more discussion). That is, tools will be able to call into Cargo in a uniform
way, with Cargo subsequently just forwarding those calls along to custom user
code hooking into an external build system. In this approach, Cargo.toml will
generally consist of a single entry forwarding to a plugin (as in the example
plugin above). The description of dependencies is then written in the external
build system's rule format. Thus, Cargo acts primarily as a *workflow and tool
orchestrator*, since it is not involved in either planning or executing the
build. Let's dig into it.

#### Workflow and interop story

Even though the external build system is entirely handling both dependency
resolution and build execution for the crates under its management, it may still
use Cargo for *lowering*, i.e. to produce the actual `rustc` invocations from a
higher-level configuration. Cargo will provide a way to do this.

When *developing* a crate, it should be possible to invoke Cargo commands as
usual. We do this via a plugin. When invoking, for example, `cargo build`, the
plugin will translate that to a request to the external build system, which will
then execute the build (possibly re-invoking Cargo for lowering). For `cargo
run`, the same steps are followed by putting the resulting build artifact in an
appropriate location, and then following Cargo's usual logic. And so on.

A similar story plays out when using, for example, the RLS or `rustfmt`. Ideally,
these tools will have no idea that a Cargo plugin is in play; the information
and artifacts they need can be obtained by using Cargo's in a standard way,
transparently -- but the underlying information will be coming from the external
build system, via the plugin. Thus the plugin for the external build system must
be able to translate its dependencies back into something equivalent to a
lockfile, at least.

### The complete picture

In general, any integration with a homogeneous build system needs to be able to
handle (vendored) crate registries, because access to crates.io is a hard constraint.

Usually, you'll want to combine the handling of these external registries with
crates managed purely by the external build system, meaning that there are
effectively *two* modes of building crates at play overall. All that's needed to
do this is a distinction within the external build system between these two
kinds of dependencies, which then drives the plugin interactions accordingly.

## Cross-cutting concern: native dependencies

One important point left out of the above explanation is the story for
dependencies on non-Rust code. These dependencies should be built and managed by
the external build system. But there's a catch: existing "sys" crates on
crates.io that provide core native dependencies use custom build scripts to
build or discover those dependencies. We want to *reroute* those crates to
instead use the dependencies provided by the build system.

Here, there's a short-term story and a long-term story.

### Short term: white lists with build script overrides

Cargo today offers the ability to [override the build script] for any crate
using the `links` key (which is generally how you signal *what* native
dependency you're providing), and instead provide the library location
directly. This feature can be used to instead point at the output provided by
the external build system. Together with whitelisting crates that use build
scripts, it's possible to use the existing crates.io ecosystem while managing
native dependencies via the external build system.

[override the build script]: http://doc.crates.io/build-script.html#overriding-build-scripts

There are some downsides, though. If the sys crates change in any way -- for
example, altering the way they build the native dependency, or the version they
use -- there's no clear heads-up that something may need to be adjusted within
the external build system. It might be possible, however, to use
version-specific whitelisting to side-step this issue.

Even so, whitelisting itself is a laborious process, and in the long run there
are advantages to offering a higher-level way of specifying native dependencies
in the first place.

### Long term: declarative native dependencies

Reliably building native dependencies in a cross-platform way
is... challenging. Today, Rust offers some help with this through crates like
[`gcc`] and [`pkgconfig`], which provide building blocks for writing build
scripts that discover or build native dependencies. But still, today, each build
script is a bespoke affair, customizing the use of these crates in arbitrary
ways. It's difficult, error-prone work.

[`gcc`]: https://docs.rs/gcc
[`pkgconfig`]: https://docs.rs/pkg-config

This RFC proposes to start a *long term* effort to provide a more first-class
way of specifying native dependencies. The hope is that we can get coverage of,
say, 80% of native dependencies using a simple, high-level specification, and
only in the remaining 20% have to write arbitrary code. And, in any case, such a
system can provide richer information about dependencies to help avoid the
downsides of the whitelisting approach.

The likely approach here is to provide [some mechanism] for using a dependency
*as* a build script, so that you could specify high-level native dependency
information directly in `Cargo.toml` attributes, and have a general tool
translate that into the appropriate build script.

[some mechanism]: https://internals.rust-lang.org/t/pre-rfc-cargo-build-and-native-dependency-integration-via-crate-based-build-scripts/5708

Needless to say, this approach will need significant experimentation. But if
successful, it would have benefits not just for build system integration, but
for using external dependencies *anywhere*.

### The story for externally-managed native dependencies

Finally, in the case where the external build system is the one specifying and
providing a native dependency, all we need is for that to result in the
appropriate flags to the lowered `rustc` invocations. If the external build
system is producing those lowered calls itself, it can completely manage this
concern. Otherwise, we will need for the plugin interface to provide a way to
plumb this information through to Cargo.

## Specifics for the mixed build system case

Switching gears, let's look at mixed build systems. Here, we may address the
need for customization with a mixture of plugins and new core Cargo
features. The primary ones on the radar right now are as follows.

- **Multiple/custom registries**. There is a longstanding desire to support
  registries other than crates.io, e.g. for private code, and to allow them to
  be used *in conjunction* with crates.io. In particular, this is a key pain
  point for customers who are otherwise happy to use Cargo as-is, but want a
  crates.io-like experience for their own code. There's
  an [RFC](https://github.com/rust-lang/rfcs/pull/2141) on this topic, and more
  work here is planned soon. Note: here, we address the needs via a
  straightforward enhancement to Cargo's features, rather than via a plugin
  system.

- **Network and source control**. We've already put significant work into
  providing control over where sources live (though vendoring) and tools for
  preventing network access. However, we could do more to make the experience
  here first class, and to give people a greater sense of control and assurance
  when using Cargo on their build farm. Here again, this is probably more about
  flags and configuration than plugins per se.

- **Caching and artifact control**. Many organizations would like to provide a
  shared build cache for the entire organization, across all of its
  projects. Here we'd likely need some kind of plugin.

These bullets are quite vague, and that's because, while we know there are needs
here, the precise problem -- let alone the solution -- it not yet clear. The
point, though, is that these are the most important problems we want to get our
head around in the foreseeable future.

## Additional areas where revisions are expected

Beyond all of the above, it seems very likely that some existing features of
Cargo will need to be revisited to fit with the build system integration
work. For example:

- **Profiles**. Putting the idea of the "build configuration" step on firmer
  footing will require clarifying the precise role of profiles, which today blur
  the line somewhat between *workflows* (e.g. `test` vs `bench`) and flags
  (e.g. `--release`). Moreover, integration with a homogeneous build system
  effectively requires that we can translate profiles on the Cargo side back and
  forth to *something* meaningful to the external build system, so that for
  example we can make `cargo test` invoke the external build system in a
  sensible way. Additional clarity here might help pave the way for [custom
  profiles] and other enhancements. On a very different note, it's not currently
  possible to control enough about the `rustc` invocation for at least some
  integration cases, and the answer may in part lie in improvements to profiles.

- **Build scripts**. Especially for homogeneous build systems, build scripts can
  pose some serious pain, because in general they may depend on numerous
  environmental factors invisibly. It may be useful to grow some ways of telling
  Cargo the precise inputs and outputs of the build script, declaratively.

- **Vendoring**. While we have [support for vendoring] dependencies today, it is
  not treated uniformly as a mirror. We may want to tighten up Cargo's
  understanding, possibly by treating vendoring in a more first-class way.

[custom profiles]: https://github.com/rust-lang/cargo/issues/2007
[support for vendoring]: https://github.com/alexcrichton/cargo-vendor/

There are undoubtedly other aspects of Cargo that will need to be touched to
achieve better build system integration; the plan as a whole is predicated on
making Cargo much more modular, which is bound to reveal concerns that should be
separated. As with everything else in this RFC, user-facing changes will require
a full RFC prior to stabilization.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This is an experimental RFC. Reference-level details will be presented in
follow-up RFCs after experimentation has concluded.

# Drawbacks
[drawbacks]: #drawbacks

It's somewhat difficult to state drawbacks for such a high-level plan; they're
more likely to arise through the particulars.

That said, it's plausible that following the plan in this RFC will result
in greater overall complexity for Cargo. The key to managing this complexity
will be ensuring that it's surfaced only on an as-needed basis. That is, uses of
Cargo in the pure crates.io ecosystem should not become more complex -- if
anything, they should become more streamlined, through improvements to features
like profiles, build scripts, and the handling of native dependencies.

# Rationale and Alternatives
[alternatives]: #alternatives

Numerous organizations we've talked to who are considering, or already are,
running Rust in production complain about difficulties with build system
integration. There's often a sense that Cargo "does too much" or is "too
opinionated", in a way that works fine for the crates.io ecosystem but is "not
realistic" when integrating into larger build systems.

It's thus critical to take steps to smooth integration, both to remove obstacles
to Rust adoption, but also to establish that Cargo has an important role to play
even within opinionated external build systems: coordinating with Rust tooling
and workflows.

This RFC is essentially a *strategic vision*, and so the alternatives are
different strategies for tackling the problem of integration. Some options
include:

- Focusing entirely on one of the use-cases mentioned. For example:
  - We could decide that it's not worthwhile to have Cargo play a role within a
  build system like [Bazel], and instead focus on users who just need to
  customize a particular aspect of Cargo. However, this would be giving up on
  the hope of providing strong integration with Rust tooling and workflows.
  - We could decide to focus solely on the [Bazel]-style use-cases. But that
    would likely push people who would otherwise be happy to use Cargo to manage
    most of their build (but need to customize some aspect) to instead try to
    manage more of the concerns themselves.

- Attempting to impose more control when integrating with hommogenous build
  systems. In the most extreme case presented above, for internal crates Cargo
  is little more than a middleman between Rust tooling and the external build
  system. We could instead support only using custom registries to manage
  crates, and hence always use Cargo's dependency resolution and so on. This
  would, however, be a non-starter for many organizations who want a
  single-version, mono-repo world internally, and it's not clear what the gains
  would be.

One key open question is: what, exactly, do Rust tools need to do their work?
Tool interop is a major goal for this effort, but ideally we'd support it with a
minimum of fuss. It may be that the needs are simple enough that we can get away
with a separate interchange format, which both Cargo and other build tools can
create. As part of the "experimental" part of this RFC, the Cargo team will work
with the Dev Tools team to fully enumerate their needs.

# Unresolved questions
[unresolved]: #unresolved-questions

Since this is an experimental RFC, there are more questions here than
answers. However, one question that would be good to tackle prior to acceptance
is: how should we prioritize various aspects of this work? Should we have any
specific customers in mind that we're trying to target (or who, better yet, are
working directly with us and plan to test and use the results)?
