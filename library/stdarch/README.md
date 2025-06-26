stdarch - Rust's standard library SIMD components
=======

[![Actions Status](https://github.com/rust-lang/stdarch/workflows/CI/badge.svg)](https://github.com/rust-lang/stdarch/actions)


# Crates

This repository contains two main crates:

* [`core_arch`](crates/core_arch/README.md) implements `core::arch` - Rust's
  core library architecture-specific intrinsics, and
  
* [`std_detect`](crates/std_detect/README.md) implements `std::detect` - Rust's
  standard library run-time CPU feature detection.

The `std::simd` component now lives in the
[`packed_simd_2`](https://github.com/rust-lang/packed_simd) crate.

## Synchronizing josh subtree with rustc

This repository is linked to `rust-lang/rust` as a [josh](https://josh-project.github.io/josh/intro.html) subtree. You can use the following commands to synchronize the subtree in both directions.

You'll need to install `josh-proxy` locally via

```
cargo install josh-proxy --git https://github.com/josh-project/josh --tag r24.10.04
```
Older versions of `josh-proxy` may not round trip commits losslessly so it is important to install this exact version.

### Pull changes from `rust-lang/rust` into this repository

1) Checkout a new branch that will be used to create a PR into `rust-lang/stdarch`
2) Run the pull command
    ```
    cargo run --manifest-path josh-sync/Cargo.toml rustc-pull
    ```
3) Push the branch to your fork and create a PR into `stdarch`

### Push changes from this repository into `rust-lang/rust`

NOTE: If you use Git protocol to push to your fork of `rust-lang/rust`,
ensure that you have this entry in your Git config,
else the 2 steps that follow would prompt for a username and password:

```
[url "git@github.com:"]
insteadOf = "https://github.com/"
```

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
GIT_CONFIG_GLOBAL=/path/to/minimal/gitconfig GIT_CONFIG_SYSTEM='' cargo run --manifest-path josh-sync/Cargo.toml -- rustc-pull
```
