- Feature Name: N/A
- Start Date: 2015-09-15
- RFC PR: [rust-lang/rfcs#1525](https://github.com/rust-lang/rfcs/pull/1525)
- Rust Issue: [rust-lang/cargo#2122](https://github.com/rust-lang/cargo/issues/2122)

# Summary

Improve Cargo's story around multi-crate single-repo project management by
introducing the concept of workspaces. All packages in a workspace will share
`Cargo.lock` and an output directory for artifacts.

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

* A workspace can contain multiple local crates: one 'root crate', and any
  number of 'member crate'.
* The root crate of a workspace has a `Cargo.toml` file containing `[workspace]`
  key, which we call it as 'root `Cargo.toml`'.
* Whenever any crate in the workspace is compiled, output will be placed in the
  `target` directory next to the root `Cargo.toml`.
* One `Cargo.lock` file for the entire workspace will reside next to the root
  `Cargo.toml` and encompass the dependencies (and dev-dependencies) for all
  crates in the workspace.

With workspaces, Cargo can now solve the problems set forth in the motivation
section. Next, however, workspaces need to be defined. In the spirit of much of
the rest of Cargo's configuration today this will largely be automatic for
conventional project layouts but will have explicit controls for configuration.

### New manifest keys

First, let's look at the new manifest keys which will be added to `Cargo.toml`:

```toml
[workspace]
members = ["relative/path/to/child1", "../child2"]

# or ...

[package]
workspace = "../foo"
```

The root `Cargo.toml` of a workspace, indicated by the presence of `[workspace]`,
is responsible for defining the entire workspace (listing all members).
This example here means that two extra crates will be members of the workspace
(which also includes the root).

The `package.workspace` key is used to point at a workspace's root crate. For
example this Cargo.toml indicates that the Cargo.toml in `../foo` is the root
Cargo.toml of root crate, that this package is a member of.

These keys are mutually exclusive when applied in `Cargo.toml`. A crate may
*either* specify `package.workspace` or specify `[workspace]`. That is, a
crate cannot both be a root crate in a workspace (contain `[workspace]`) and
also be a member crate of another workspace (contain `package.workspace`).

### "Virtual" `Cargo.toml`

A good number of projects do not necessarily have a "root `Cargo.toml`" which is
an appropriate root for a workspace. To accommodate these projects and allow for
the output of a workspace to be configured regardless of where crates are
located, Cargo will now allow for "virtual manifest" files. These manifests will
currently **only** contains the `[workspace]` table and will notably be lacking
a `[project]` or `[package]` top level key.

Cargo will for the time being disallow many commands against a virtual manifest,
for example `cargo build` will be rejected. Arguments that take a package,
however, such as `cargo test -p foo` will be allowed. Workspaces can eventually
get extended with `--all` flags so in a workspace root you could execute
`cargo build --all` to compile all crates.

### Validating a workspace

A workspace is valid if these two properties hold:

1. A workspace has only one root crate (that with `[workspace]` in
   `Cargo.toml`).
2. All workspace crates defined in `workspace.members` point back to the
   workspace root with `package.workspace`.

While the restriction of one-root-per workspace may make sense, the restriction
of crates pointing back to the root may not. If, however, this restriction were
not in place then the set of crates in a workspace may differ depending on
which crate it was viewed from. For example if workspace root A includes B then
it will think B is in A's workspace. If, however, B does not point back to A,
then B would not think that A was in its workspace. This would in turn cause the
set of crates in each workspace to be different, further causing `Cargo.lock` to
get out of sync if it were allowed. By ensuring that all crates have edges to
each other in a workspace Cargo can prevent this situation and guarantee robust
builds no matter where they're executed in the workspace.

To alleviate misconfiguration Cargo will emit an error if the two properties
above do not hold for any crate attempting to be part of a workspace. For
example, if the `package.workspace` key is specified, but the crate is not a
workspace root or doesn't point back to the original crate an error is emitted.

### Implicit relations

The combination of the `package.workspace` key and `[workspace]` table is enough
to specify any workspace in Cargo. Having to annotate all crates with a
`package.workspace` parent or a `workspace.members` list can get quite tedious,
however! To alleviate this configuration burden Cargo will allow these keys to
be implicitly defined in some situations.

The `package.workspace` can be omitted if it would only contain `../` (or some
repetition of it). That is, if the root of a workspace is hierarchically the
first `Cargo.toml` with `[workspace]` above a crate in the filesystem, then that
crate can omit the `package.workspace` key.

Next, a crate which specifies `[workspace]` **without a `members` key** will
transitively crawl `path` dependencies to fill in this key. This way all `path`
dependencies (and recursively their own `path` dependencies) will inherently
become the default value for `workspace.members`.

Note that these implicit relations will be subject to the same validations
mentioned above for all of the explicit configuration as well.

### Workspaces in practice

Many Rust projects today already have `Cargo.toml` at the root of a repository,
and with the small addition of `[workspace]` in the root `Cargo.toml`, a
workspace will be ready for all crates in that repository. For example:

* An FFI crate with a sub-crate for FFI bindings

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
  members = ["crate1", "crate2", "crate3"]
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
  members = ["crate1", "crate2"]
  ```

  ```toml
  # ws2/Cargo.toml
  [workspace]
  ```

* Trees with non-hierarchical workspaces

  ```
  root/
    Cargo.toml
    src/
  crates/
    crate1/
      Cargo.toml
      src/
    crate2/
      Cargo.toml
      src/
  ```

  The workspace here can be configured by placing the following in the
  manifests:

  ```toml
  # root/Cargo.toml
  #
  # Note that `members` aren't necessary if these are otherwise path
  # dependencies.
  [workspace]
  members = ["../crates/crate1", "../crates/crate2"]
  ```

  ```toml
  # crates/crate1/Cargo.toml
  [package]
  workspace = "../root"
  ```

  ```toml
  # crates/crate2/Cargo.toml
  [package]
  workspace = "../root"
  ```

Projects like the compiler will likely need exhaustively explicit configuration.
The `rust` repo conceptually has two workspaces, the standard library and the
compiler, and these would need to be manually configured with
`workspace.members` and `package.workspace` keys amongst all crates.

### Lockfile and override interactions

One of the main features of a workspace is that only one `Cargo.lock` is
generated for the entire workspace. This lock file can be affected, however,
with both [`[replace]` overrides][replace] as well as `paths` overrides.

[replace]: https://github.com/rust-lang/cargo/pull/2385

Primarily, the `Cargo.lock` generate will not simply be the concatenation of the
lock files from each project. Instead the entire workspace will be resolved
together all at once, minimizing versions of crates used and sharing
dependencies as much as possible. For example one `path` dependency will always
have the same set of dependencies no matter which crate is being compiled.

When interacting with overrides, workspaces will be modified to only allow
`[replace]` to exist in the workspace root. This Cargo.toml will affect lock
file generation, but no other workspace members will be allowed to have a
`[replace]` directive (with an informative error message being produced).

Finally, the `paths` overrides will be applied as usual, and they'll continue to
be applied relative to whatever crate is being compiled (not the workspace
root). These are intended for much more local testing, so no restriction of
"must be in the root" should be necessary.

Note that this change to the lockfile format is technically incompatible with
older versions of Cargo.lock, but the entire workspaces feature is also
incompatible with older versions of Cargo. This will require projects that wish
to work with workspaces and multiple versions of Cargo to check in multiple
`Cargo.lock` files, but if projects avoid workspaces then Cargo will remain
forwards and backwards compatible.

### Future Extensions

Once Cargo understands a workspace of crates, we could easily extend various
subcommands with a `--all` flag to perform tasks such as:

* Test all crates within a workspace (run all unit tests, doc tests, etc)
* Build all binaries for a set of crates within a workspace
* Publish all crates in a workspace if necessary to crates.io

Furthermore, workspaces could start to deduplicate metadata among crates like
version numbers, URL information, authorship, etc.

This support isn't proposed to be added in this RFC specifically, but simply to
show that workspaces can be used to solve other existing issues in Cargo.

# Drawbacks

* As proposed there is no method to disable implicit actions taken by Cargo.
  It's unclear what the use case for this is, but it could in theory arise.

* No crate will implicitly benefit from workspaces after this is implemented.
  Existing crates must opt-in with a `[workspace]` key somewhere at least.

# Alternatives

* The `workspace.members` key could support globs to define a number of
  directories at once. For example one could imagine:

  ```toml
  [workspace]
  members = ["crates/*"]
  ```

  as an ergonomic method of slurping up all sub-folders in the `crates` folder
  as crates.

* Cargo could attempt to perform more inference of workspace members by simply
  walking the entire directory tree starting at `Cargo.toml`. All children found
  could implicitly be members of the workspace. Walking entire trees,
  unfortunately, isn't always efficient to do and it would be unfortunate to
  have to unconditionally do this.

# Unresolved questions

* Does this approach scale well to repositories with a large number of crates?
  For example does the winapi-rs repository experience a slowdown on standard
  `cargo build` as a result?
