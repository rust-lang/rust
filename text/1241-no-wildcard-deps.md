- Feature Name: N/A
- Start Date: 2015-07-23
- RFC PR: [rust-lang/rfcs#1241](https://github.com/rust-lang/rfcs/pull/1241)
- Rust Issue: [rust-lang/rust#28628](https://github.com/rust-lang/rust/issues/28628)

# Summary

A Cargo crate's dependencies are associated with constraints that specify the
set of versions of the dependency with which the crate is compatible. These
constraints range from accepting exactly one version (`=1.2.3`), to
accepting a range of versions (`^1.2.3`, `~1.2.3`, `>= 1.2.3, < 3.0.0`), to
accepting any version at all (`*`). This RFC proposes to update crates.io to
reject publishes of crates that have compile or build dependencies with
a wildcard version constraint.

# Motivation

Version constraints are a delicate balancing act between stability and
flexibility. On one extreme, one can lock dependencies to an exact version.
From one perspective, this is great, since the dependencies a user will consume
will be the same that the developers tested against. However, on any nontrival
project, one will inevitably run into conflicts where library A depends on
version `1.2.3` of library B, but library C depends on version `1.2.4`, at
which point, the only option is to force the version of library B to one of
them and hope everything works.

On the other hand, a wildcard (`*`) constraint will never conflict with
anything! There are other things to worry about here, though. A version
constraint is fundamentally an assertion from a library's author to its users
that the library will work with any version of a dependency that matches its
constraint. A wildcard constraint is claiming that the library will work with
any version of the dependency that has ever been released *or will ever be
released, forever*. This is a somewhat absurd guarantee to make - forever is a
long time!

Absurd guarantees on their own are not necessarily sufficient motivation to
make a change like this. The real motivation is the effect that these
guarantees have on consumers of libraries.

As an example, consider the [openssl](https://crates.io/crates/openssl) crate.
It is one of the most popular libraries on crates.io, with several hundred
downloads every day. 50% of the [libraries that depend on it](https://crates.io/crates/openssl/reverse_dependencies)
have a wildcard constraint on the version. None of them can build against every
version that has ever been released. Indeed, no libraries can since many of
those releases can before Rust 1.0 released. In addition, almost all of them
them will fail to compile against version 0.7 of openssl when it is released.
When that happens, users of those libraries will be forced to manually override
Cargo's version selection every time it is recalculated. This is not a fun
time.

Bad version restrictions are also "viral". Even if a developer is careful to
pick dependencies that have reasonable version restrictions, there could be a
wildcard constraint hiding five transitive levels down.  Manually searching the
entire dependency graph is an exercise in frustration that shouldn't be
necessary.

On the other hand, consider a library that has a version constraint of `^0.6`.
When openssl 0.7 releases, the library will either continue to work against
version 0.7, or it won't. In the first case, the author can simply extend the
constraint to `>= 0.6, < 0.8` and consumers can use it with version 0.6 or 0.7
without any trouble. If it does not work against version 0.7, consumers of the
library are fine! Their code will continue to work without any manual
intervention. The author can update the library to work with version 0.7 and
release a new version with a constraint of `^0.7` to support consumers that
want to use that newer release.

Making crates.io more picky than Cargo itself is not a new concept; it
currently [requires several items](https://github.com/rust-lang/crates.io/blob/8c85874b6b967e1f46ae2113719708dce0c16d32/src/krate.rs#L746-L759) in published crates that Cargo will not:

 * A valid license
 * A description
 * A list of authors

All of these requirements are in place to make it easier for developers to use
the libraries uploaded to crates.io - that's why crates are published, after
all! A restriction on wildcards is another step down that path.

Note that this restriction would only apply to normal compile dependencies and
build dependencies, but not to dev dependencies. Dev dependencies are only used
when testing a crate, so it doesn't matter to downstream consumers if they
break.

This RFC is not trying to prohibit *all* constraints that would run into the
issues described above. For example, the constraint `>= 0.0.0` is exactly
equivalent to `*`. This is for a couple of reasons:

* It's not totally clear how to precisely define "reasonable" constraints. For
example, one might want to forbid constraints that allow unreleased major
versions. However, some crates provide strong guarantees that any breaks will
be followed by one full major version of deprecation. If a library author is
sure that their crate doesn't use any deprecated functionality of that kind of
dependency, it's completely safe and reasonable to explicitly extend the
version constraint to include the next unreleased version.
* Cargo and crates.io are missing tools to deal with overly-restrictive
constraints. For example, it's not currently possible to force Cargo to allow
dependency resolution that violates version constraints. Without this kind of
support, it is somewhat risky to push too hard towards tight version
constraints.
* Wildcard constraints are popular, at least in part, because they are the
path of least resistance when writing a crate. Without wildcard constraints,
crate authors will be forced to figure out what kind of constraints make the
most sense in their use cases, which may very well be good enough.

# Detailed design

The prohibition on wildcard constraints will be rolled out in stages to make
sure that crate authors have lead time to figure out their versioning stories.

In the next stable Rust release (1.4), Cargo will issue warnings for all
wildcard constraints on build and compile dependencies when publishing, but
publishes those constraints will still succeed. Along side the next stable
release after that (1.5 on December 11th, 2015), crates.io be updated to reject
publishes of crates with those kinds of dependency constraints. Note that the
check will happen on the crates.io side rather than on the Cargo side since
Cargo can publish to locations other than crates.io which may not worry about
these restrictions.

# Drawbacks

The barrier to entry when publishing a crate will be mildly higher.

Tightening constraints has the potential to cause resolution breakage when no
breakage would occur otherwise.

# Alternatives

We could continue allowing these kinds of constraints, but complain in a
"sufficiently annoying" manner during publishes to discourage their use.

This RFC originally proposed forbidding all constraints that had no upper
version bound but has since been pulled back to just `*` constraints.
