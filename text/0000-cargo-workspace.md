- Feature Name: N/A
- Start Date: 2015-09-15
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Improve Cargo's story around multi-crate single-repo project management by
introducing the concept of workspaces. All packages in a workspace will share
`Cargo.lock` and an output directory for artifacts.

Cargo will infer workspaces where possible, but it will also have knobs for
explicitly controlling what crates belong to which workspace.

# Motivation

A common method to organize a multi-crate project is to have one
repository which contains all of the crates. Each crate has a corresponding
subdirectory along with a `Cargo.toml` describing how to build it. There are a
number of downsides to this approach, however:

* Each sub-crate will have its own `Cargo.lock`, so it's difficult to ensure
  that the entire project is using the same version of all dependencies. This is
  desired as the main crate (often a binary) is often the one that has the
  `Cargo.lock` "which counts", but it needs to be kept in sync with all
  dependencies.

* When building or testing sub-crates, all dependencies will be recompiled as
  the target directory will be changing as you move around the source tree. This
  can be overridden with `build.target-dir` or `CARGO_TARGET_DIR`, but this
  isn't always convenient to set.

Solving these two problems should help ease the development of large Rust
projects by ensuring that all dependencies remain in sync and builds by default
use already-built artifacts if available.

# Detailed design

Cargo will grow the concept of a **workspace** for managing repositories of
multiple crates. Workspaces will then have the properties:

* A workspace can contain multiple local crates.
* Each workspace will have a root.
* Whenever any crate in the workspace is compiled, output will be placed in the
  `target` directory next to the root.
* One `Cargo.lock` for the entire workspace will reside next to the workspace
  root and encompass the dependencies (and dev-dependencies) for all packages
  in the workspace.

With workspaces, Cargo can now solve the problems set forth in the motivation
section. Next, however, workspaces need to be defined. In the spirit of much of
the rest of Cargo's configuration today this will largely be automatic for
conventional project layouts but will have explicit controls for configuration.

### New manifest keys

First, let's look at the new manifest keys which will be added to `Cargo.toml`:

```toml
[workspace]
root = true
members = ["relative/path/to/child1", "child2"]
```

Here the `workspace.root` key will be used to indicate whether a `Cargo.toml` is
the root of a workspace, and the `members` key will be a list of paths to
crates which should be added to the package's workspace. The paths listed in
`members` must be valid paths to crates.

### Implicit relations

In addition to the keys above, Cargo will apply a few heuristics to infer the
keys wherever possible:

* All path dependencies of a crate are considered members of the `workspace` key
  implicitly.
* Starting from a package's `Cargo.toml`, Cargo will walk upwards on the
  filesystem to find a sibling `Cargo.toml` and VCS directory (e.g. `.git` or
  `.svn`). If found, this crate is also implicitly considered a member of the
  workspace.
* A `Cargo.toml` which resides next to a VCS directory is implicitly a
  workspace root.

These rules are intended to reflect some conventional Cargo project layouts.
"Root crates" typically appear at the root of a repository with lots path
dependencies to all other crates in a repo. Additionally, we don't want to
traverse wildly across the filesystem so we only go upwards to a fixed point or
downwards to specific locations.

### "Virtual" `Cargo.toml`

A good number of projects do not have a root `Cargo.toml` at the top of a
repository, however. While the explicit `[workspace]` keys should be enough to
configure the workspace in addition to the implicit relations above, this
directory structure is common enough that it shouldn't require *that* much more
configuration.

To accomodate this project layout, Cargo will now allow for "virtual manifest"
files. These manifests will currently **only** contains the `[workspace]` key
and will notably be lacking a `[project]` or `[package]` top level key.

A virtual manifest does not itself define a crate, but can help when defining a
root. For example a `Cargo.toml` file at the root of a repository with
`workspace.members` keys would suffice for the project configurations in
question.

Cargo will for the time being disallow many commands against a virtual manifest,
for example `cargo build` will be rejected. Arguments that take a package,
however, such as `cargo test -p foo` will be allowed. Workspaces can eventually
get extended with `--all` flags so in a workspace root you could execute
`cargo build --all` to compile all crates.

### Constructing a workspace

With the explicit and implicit relations defined above, each crate will now have
a flag indicating whether it's the root and a number of outgoing edges to other
crates. Two crates are then in the same workspace if they both transitively have
edges to one another. A valid workspace then only has one crate that is a root.

While the restriction of one-root-per workspace may make sense, the restriction
of crates transitively having edges to one another may seem a bit odd. The
intention is to ensure that the set of packages in a workspace is the same
regardless of which package is selected to start discovering a workspace from.

With the implicit relations defined it's possible for a repository to not have a
root package yet still have path dependencies. In this situation each dependency
would not know how to get back to the "root package", so the workspace from the
point of view of the path dependencies would be different than that of the root
package. This could in turn lead to `Cargo.lock` getting out of sync.

To alleviate misconfiguration, however, if the `workspace` configuration key
contains a crate which is not a member of the constructed workspace, Cargo will
emit an error indicating such.

### Workspaces in practice

The conventional layout for a Rust project is to have a `Cargo.toml` at the root
with the "main project" with dependencies and/or satellite projects underneath.
Consequently the conventional layout will need no extra configuration to benefit
from the workspaces proposed in this RFC. For example, all of these project
layouts (with `/` being the root of a repository) will not require any
configuration to have all crates be members of a workspace:

* An FFI crate with a sub-scrate for FFI bindings

  ```
  Cargo.toml
  src/
  foo-sys/
    Cargo.toml
    src/
  ```

* A crate with multiple in-tree dependencies

  ```
  Cargo.toml
  src/
  dep1/
    Cargo.toml
    src/
  dep2/
    Cargo.toml
    src/
  ```

Projects like the compiler, however, will likely need explicit configuration.
The `rust` repo conceptually has two workspaces, the standard library and the
compiler, and these would need to be manually configured with `workspace` and
`workspace-root` keys amongst all crates.

Some examples of layouts that will require extra configuration, along with the
configuration necessary, are:

* Trees without any root crate

  ```
  crate1/
    Cargo.toml
    src/
  crate2/
    Cargo.toml
    src/
  crate3/
    Cargo.toml
    src/
  ```

  these crates can all join the same workspace via a `Cargo.toml` file at the
  root looking like:

  ```toml
  [workspace]
  members = [
    "crate1",
    "crate2",
    "crate3",
  ]
  ```

* Trees with multiple workspaces

  ```
  ws1/
    crate1/
      Cargo.toml
      src/
    crate2/
      Cargo.toml
      src/
  ws2/
    Cargo.toml
    src/
    crate3/
      Cargo.toml
      src/
  ```

  The two workspaces here can be configured by placing the following in the
  manifests:

  ```toml
  # ws1/Cargo.toml
  [workspace]
  root = true
  members = ["crate1", "crate2"]
  ```

  ```toml
  # ws1/crate1/Cargo.toml
  [workspace]
  members = [".."]
  ```

  ```toml
  # ws1/crate2/Cargo.toml
  [workspace]
  members = [".."]
  ```

  ```toml
  # ws2/Cargo.toml
  [workspace]
  root = true
  ```

  ```toml
  # ws2/crate3/Cargo.toml
  [workspace]
  members = [".."]
  ```

### Future Extensions

Once Cargo understands a workspace of crates, we could easily extend various
subcommands with a `--all` flag to perform tasks such as:

* Test all crates within a workspace (run all unit tests, doc tests, etc)
* Build all binaries for a set of crates within a workspace
* Publish all crates in a workspace if necessary to crates.io

This support isn't proposed to be added in this RFC specifically, but simply to
show that workspaces can be used to solve other existing issues in Cargo.

# Drawbacks

* This change is not backwards compatible with older versions of Cargo.lock. For
  example if a newer cargo were used to develop a repository which otherwise is
  developed with older versions of Cargo, the `Cargo.lock` files generated would
  be incompatible. If all maintainers agree on versions of Cargo, however, this
  is not a problem.

* If no crate exists at the root of a repository, it may be the case that an
  unduly large amount of configuration is required to setup the workspace
  correctly. A minor deviation from the normal conventions should in theory only
  require a proportionally minor amount of configuration.

* As proposed there is no method to disable implicit actions taken by Cargo.
  It's unclear what the use case for this is, but it could in theory arise.

# Alternatives

* Cargo could attempt to perform more inference of workspace members by simply
  walking the entire directory tree starting at `Cargo.toml`. All children found
  could implicitly be members of the workspace. Walking entire trees,
  unfortunately, isn't always efficient to do and it would be unfortunate to
  have to unconditionally do this.

* Cargo could support "virtual packages" where a `Cargo.toml` is placed at the
  root of a repository but only to serve as a global project configuration. No
  crate would actually be described by a virtual package, but it would play into
  the workspace heuristics described here. This feature could alleviate the "too
  much extra configuration" drawback described above, but it's unclear whether
  it's needed at this point.

* Implicit members are currently only path dependencies and a "Cargo.toml next
  to VCS" traveling upwards. Instead all Cargo.toml members found traveling
  upwards could be implicit members of a workspace. This behavior, however, may
  end up picking up too many crates.

# Unresolved questions

* Does this approach scale well to repositories with a large number of crates?
  For example does the winapi-rs repository experience a slowdown on standard
  `cargo build` as a result?
