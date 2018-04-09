- Feature Name: `metabuild`
- Start Date: 2017-10-31
- RFC PR: [rust-lang/rfcs#2196](https://github.com/rust-lang/rfcs/pull/2196)
- Rust Issue: [rust-lang/rust#49803](https://github.com/rust-lang/rust/issues/49803)

# Summary

Introduce a mechanism for Cargo crates to make use of declarative build
scripts, obtained from one or more of their dependencies rather than via a
`build.rs` file. Support experimentation with declarative build scripts in the
crates.io ecosystem.

# Motivation

Cargo has many potentially desirable enhancements planned for its build
process, including integrating a Cargo build process with native dependencies,
and integrating with broader build systems or projects, such as massive
mono-repo build systems, or Linux distributions.

Right now, the biggest problem facing such systems involves `build.rs` scripts
and the arbitrary things those scripts can do. Such build systems typically
need more information about native dependencies that are embedded in
`build.rs`, so that they can provide their own versions of those dependencies,
or encode appropriate dependencies in another metadata format such as the
dependencies of their packaging system or build system. Right now, such systems
often have to override the `build.rs` script themselves, and do custom
per-crate integration work, manually; there’s no way to introspect what
`build.rs` does, or get a declarative semantic description of the build script.

At the same time, we don't yet have sufficiently precise information about the
needs of such systems to design an ideal set of Cargo metadata on the first
try. Rather than attempt to architect the perfect solution from the start, and
potentially create an intermediate state that will require long-term support,
we propose to allow experimentation with declarative build systems within the
crates.io ecosystem, in crates supplying modular components similar to
`build.rs` scripts. By convention, such scripts should typically read any
parameters and metadata they need from `Cargo.toml`, in a form that other
build-related software can read as well.

# Guide-level explanation

In the `[package]` section of `Cargo.toml`, you can specify a field
`metabuild`, whose value should be a string or list of strings, each one
exactly matching the name of a dependency specified in the
`[build-dependencies]` section. If you specify `metabuild`, you must not
specify `build`, and Cargo will ignore the `build.rs` file if any.

When Cargo builds a crate that specifies a `metabuild` field, at the point when
it would have built and run `build.rs`, it will instead invoke the
`metabuild()` function from each of the specified crates in order.

In effect, Cargo will act as though it had a `build.rs` file containing an
`extern crate` line for each string, in order, as well as a `main` function
that calls the `metabuild` function in each such crate, in order. For example,
if the crate contains `metabuild = ["pkgc", "parsegen"]`, then the effective
`build.rs` will look like this:

```rust
extern crate pkgc;
extern crate parsegen;

fn main() {
    pkgc::metabuild();
    parsegen::metabuild();
}
```

Note that the `metabuild` functions intentionally take no parameters; they
should obtain any parameters they need from `Cargo.toml`. Various crates to
parse `Cargo.toml` exist in the crates.io ecosystem.

Also note that the `metabuild` functions do not return an error type; if they
fail, they should panic.

Future versions of this interface with higher integration into Cargo may
incorporate ways for Cargo to pass pre-parsed data from `Cargo.toml`, or ways
for the `metabuild` functions to return semantic error information. Metabuild
interfaces may also wish to run scripts in parallel, provide dependencies
between them, or orchestrate their execution in many other ways. This minimal
specification allows for experimentation with such interfaces within the
crates.io ecosystem, by providing an adapter from the raw metabuild interface.

# Reference-level explanation

Cargo's logic to invoke `build.rs` should check for the `metabuild` key, and if
present, create and invoke a temporary `build.rs` as described above. For an
initial implementation, Cargo can generate and cache that `build.rs` in the
`target` directory when needed, alongside the built version of the script.

For Cargo schema versioning, using the `metabuild` key will result in the crate
requiring a sufficiently new version of Cargo to understand `metabuild`. This
should start out as an unstable Cargo feature; in the course of experimentation
and stabilization, the implementation of this feature may change, requiring
adaptation of experimental build scripts.

If any of the strings mentioned in `metabuild` do not match one of the
build-dependencies, Cargo should produce an error (*before* attempting to
generate and compile a `build.rs` script). However, if a string matches a
conditional build-dependency, such as one conditional on a feature or target,
then Cargo should only invoke that build-dependency's `metabuild` function when
those conditions apply.

Cargo's documentation on `metabuild` should recommend a preferred crate for
parsing data from `Cargo.toml`, to avoid every provider of a metabuild function
from reimplementing it themselves.

As we develop other best practices for the development and implementation of
metabuild crates, we should extract and standardize common code for those
practices as crates.

# Drawbacks

While Cargo can change this interface arbitrarily while still unstable, one
stabilized, Cargo will have to support it forever, even if we develop a new
build/metabuild interface in the future.

# Rationale and Alternatives

`metabuild` could always point to a single crate, and not support a list of
crate names; a crate in the crates.io ecosystem could easily provide the "list
of crate names" functionality, along with more advanced flows of information
from one such crate to another. However, many simple cases will only want to
invoke a list of crates in order, and handling that one case within Cargo will
simplify initial experimentation while still allowing implementation of more
complex logic via other crates in the crates.io ecosystem.

`metabuild()` functions could take parameters, return errors, or make use of
traits. However, this would require providing appropriate types and traits for
all of those, as well as a helper crate providing those types and traits, and
we do not yet know what interfaces we need or want. We propose experimenting
via the crates.io ecosystem first, before considering such interfaces.

Cargo could compile and run a separate `build.rs`-like script to run each
metabuild function independently, rather than a single script that invokes all
of them.

We could avoid introducing an extensible mechanism, and instead introduce
individual semantic build interfaces one-by-one within Cargo itself. However,
this would drastically impair experimentation and development, and in
particular this would make it more difficult to evaluate multiple potential
approaches to any given piece of build functionality. Such an interface would
also not provide an obvious path to support code generators.
