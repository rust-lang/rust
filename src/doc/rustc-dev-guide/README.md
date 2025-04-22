[![CI](https://github.com/rust-lang/rustc-dev-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-lang/rustc-dev-guide/actions/workflows/ci.yml)


This is a collaborative effort to build a guide that explains how rustc
works. The aim of the guide is to help new contributors get oriented
to rustc, as well as to help more experienced folks in figuring out
some new part of the compiler that they haven't worked on before.

[You can read the latest version of the guide here.](https://rustc-dev-guide.rust-lang.org/)

You may also find the rustdocs [for the compiler itself][rustdocs] useful.
Note that these are not intended as a guide; it's recommended that you search
for the docs you're looking for instead of reading them top to bottom.

[rustdocs]: https://doc.rust-lang.org/nightly/nightly-rustc

For documentation on developing the standard library, see
[`std-dev-guide`](https://std-dev-guide.rust-lang.org/).

### Contributing to the guide

The guide is useful today, but it has a lot of work still to go.

If you'd like to help improve the guide, we'd love to have you! You can find
plenty of issues on the [issue
tracker](https://github.com/rust-lang/rustc-dev-guide/issues). Just post a
comment on the issue you would like to work on to make sure that we don't
accidentally duplicate work. If you think something is missing, please open an
issue about it!

**In general, if you don't know how the compiler works, that is not a
problem!** In that case, what we will do is to schedule a bit of time
for you to talk with someone who **does** know the code, or who wants
to pair with you and figure it out.  Then you can work on writing up
what you learned.

In general, when writing about a particular part of the compiler's code, we
recommend that you link to the relevant parts of the [rustc
rustdocs][rustdocs].

### Build Instructions

To build a local static HTML site, install [`mdbook`](https://github.com/rust-lang/mdBook) with:

```
cargo install mdbook mdbook-linkcheck2 mdbook-toc mdbook-mermaid
```

and execute the following command in the root of the repository:

```
mdbook build --open
```

The build files are found in the `book/html` directory.

### Link Validations

We use `mdbook-linkcheck2` to validate URLs included in our documentation. Link
checking is **not** run by default locally, though it is in CI. To enable it
locally, set the environment variable `ENABLE_LINKCHECK=1` like in the
following example.

```
ENABLE_LINKCHECK=1 mdbook serve
```

### Table of Contents

We use `mdbook-toc` to auto-generate TOCs for long sections. You can invoke the preprocessor by
including the `<!-- toc -->` marker at the place where you want the TOC.

## Synchronizing josh subtree with rustc

This repository is linked to `rust-lang/rust` as a [josh](https://josh-project.github.io/josh/intro.html) subtree. You can use the following commands to synchronize the subtree in both directions.

You'll need to install `josh-proxy` locally via

```
cargo +stable install josh-proxy --git https://github.com/josh-project/josh --tag r24.10.04
```
Older versions of `josh-proxy` may not round trip commits losslessly so it is important to install this exact version.

### Pull changes from `rust-lang/rust` into this repository

1) Checkout a new branch that will be used to create a PR into `rust-lang/rustc-dev-guide`
2) Run the pull command
    ```
    cargo run --manifest-path josh-sync/Cargo.toml rustc-pull
    ```
3) Push the branch to your fork and create a PR into `rustc-dev-guide`

### Push changes from this repository into `rust-lang/rust`
1) Run the push command to create a branch named `<branch-name>` in a `rustc` fork under the `<gh-username>` account
    ```
    cargo run --manifest-path josh-sync/Cargo.toml rustc-push <branch-name> <gh-username>
    ```
2) Create a PR from `<branch-name>` into `rust-lang/rust`

#### Minimal git config

For simplicity (ease of implementation purposes), the josh-sync script simply calls out to system git. This means that the git invocation may be influenced by global (or local) git configuration.

You may observe "Nothing to pull" even if you *know* rustc-pull has something to pull if your global git config sets `fetch.prunetags = true` (and possibly other configurations may cause unexpected outcomes).

To minimize the likelihood of this happening, you may wish to keep a separate *minimal* git config that *only* has `[user]` entries from global git config, then repoint system git to use the minimal git config instead. E.g.

```
GIT_CONFIG_GLOBAL=/path/to/minimal/gitconfig GIT_CONFIG_SYSTEM='' cargo +stable run --manifest-path josh-sync/Cargo.toml -- rustc-pull
```
