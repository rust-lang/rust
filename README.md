# Rust - wasix

This is a fork of [Rust](https://github.com/rust-lang/rust) with support for
the `wasix`, a Webassembly target that is a superset of `wasi`, extended with
additional functionality.

See [wasix.org](https://wasix.org) for more details.

## Usage

To compile Rust code to `wasix`, you should usually use the
[cargo-wasix](https://github.com/wasix-org/cargo-wasix) cargo wrapper.

It will automatically download pre-built versions of the wasix Rust toolchain,
and handles automating the integration with cargo.

See the above repository for more information.
