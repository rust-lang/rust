# Installation

If you're using `rustup` to install and manage your Rust toolchains, Clippy is
usually **already installed**. In that case you can skip this chapter and go to
the [Usage] chapter.

> Note: If you used the `minimal` profile when installing a Rust toolchain,
> Clippy is not automatically installed.

## Using Rustup

If Clippy was not installed for a toolchain, it can be installed with

```
$ rustup component add clippy [--toolchain=<name>]
```

## From Source

Take a look at the [Basics] chapter in the Clippy developer guide to find step
by step instructions on how to build and install Clippy from source.

[Basics]: development/basics.md#install-from-source
[Usage]: usage.md
