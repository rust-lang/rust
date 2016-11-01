libc
====

A Rust library with native bindings to the types and functions commonly found on
various systems, including libc.

[![Build Status](https://travis-ci.org/rust-lang/libc.svg?branch=master)](https://travis-ci.org/rust-lang/libc)
[![Build status](https://ci.appveyor.com/api/projects/status/34csq3uurnw7c0rl?svg=true)](https://ci.appveyor.com/project/alexcrichton/libc)

[Documentation](#platforms-and-documentation)

## Usage

First, add the following to your `Cargo.toml`:

```toml
[dependencies]
libc = "0.2"
```

Next, add this to your crate root:

```rust
extern crate libc;
```

Currently libc by default links to the standard library, but if you would
instead like to use libc in a `#![no_std]` situation or crate you can request
this via:

```toml
[dependencies]
libc = { version = "0.2", default-features = false }
```

## What is libc?

The primary purpose of this crate is to provide all of the definitions necessary
to easily interoperate with C code (or "C-like" code) on each of the platforms
that Rust supports. This includes type definitions (e.g. `c_int`), constants
(e.g. `EINVAL`) as well as function headers (e.g. `malloc`).

This crate does not strive to have any form of compatibility across platforms,
but rather it is simply a straight binding to the system libraries on the
platform in question.

## Public API

This crate exports all underlying platform types, functions, and constants under
the crate root, so all items are accessible as `libc::foo`. The types and values
of all the exported APIs match the platform that libc is compiled for.

More detailed information about the design of this library can be found in its
[associated RFC][rfc].

[rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1291-promote-libc.md

## Adding an API

Want to use an API which currently isn't bound in `libc`? It's quite easy to add
one!

The internal structure of this crate is designed to minimize the number of
`#[cfg]` attributes in order to easily be able to add new items which apply
to all platforms in the future. As a result, the crate is organized
hierarchically based on platform. Each module has a number of `#[cfg]`'d
children, but only one is ever actually compiled. Each module then reexports all
the contents of its children.

This means that for each platform that libc supports, the path from a
leaf module to the root will contain all bindings for the platform in question.
Consequently, this indicates where an API should be added! Adding an API at a
particular level in the hierarchy means that it is supported on all the child
platforms of that level. For example, when adding a Unix API it should be added
to `src/unix/mod.rs`, but when adding a Linux-only API it should be added to
`src/unix/notbsd/linux/mod.rs`.

If you're not 100% sure at what level of the hierarchy an API should be added
at, fear not! This crate has CI support which tests any binding against all
platforms supported, so you'll see failures if an API is added at the wrong
level or has different signatures across platforms.

With that in mind, the steps for adding a new API are:

1. Determine where in the module hierarchy your API should be added.
2. Add the API.
3. Send a PR to this repo.
4. Wait for CI to pass, fixing errors.
5. Wait for a merge!

### Test before you commit

We have two automated tests running on [Travis](https://travis-ci.org/rust-lang/libc):

1. [`libc-test`](https://github.com/alexcrichton/ctest)
  - `cd libc-test && cargo run`
  - Use the `skip_*()` functions in `build.rs` if you really need a workaround.
2. Style checker
  - `rustc ci/style.rs && ./style src`

## Platforms and Documentation

The following platforms are currently tested and have documentation available:

Tested:
  * [`i686-pc-windows-msvc`](https://doc.rust-lang.org/libc/i686-pc-windows-msvc/libc/)
  * [`x86_64-pc-windows-msvc`](https://doc.rust-lang.org/libc/x86_64-pc-windows-msvc/libc/)
    (Windows)
  * [`i686-pc-windows-gnu`](https://doc.rust-lang.org/libc/i686-pc-windows-gnu/libc/)
  * [`x86_64-pc-windows-gnu`](https://doc.rust-lang.org/libc/x86_64-pc-windows-gnu/libc/)
  * [`i686-apple-darwin`](https://doc.rust-lang.org/libc/i686-apple-darwin/libc/)
  * [`x86_64-apple-darwin`](https://doc.rust-lang.org/libc/x86_64-apple-darwin/libc/)
    (OSX)
  * `i686-apple-ios`
  * `x86_64-apple-ios`
  * [`i686-unknown-linux-gnu`](https://doc.rust-lang.org/libc/i686-unknown-linux-gnu/libc/)
  * [`x86_64-unknown-linux-gnu`](https://doc.rust-lang.org/libc/x86_64-unknown-linux-gnu/libc/)
    (Linux)
  * [`x86_64-unknown-linux-musl`](https://doc.rust-lang.org/libc/x86_64-unknown-linux-musl/libc/)
    (Linux MUSL)
  * [`aarch64-unknown-linux-gnu`](https://doc.rust-lang.org/libc/aarch64-unknown-linux-gnu/libc/)
  * [`mips-unknown-linux-gnu`](https://doc.rust-lang.org/libc/mips-unknown-linux-gnu/libc/)
  * [`arm-unknown-linux-gnueabihf`](https://doc.rust-lang.org/libc/arm-unknown-linux-gnueabihf/libc/)
  * [`arm-linux-androideabi`](https://doc.rust-lang.org/libc/arm-linux-androideabi/libc/)
    (Android)
  * [`x86_64-unknown-freebsd`](https://doc.rust-lang.org/libc/x86_64-unknown-freebsd/libc/)
  * [`x86_64-unknown-openbsd`](https://doc.rust-lang.org/libc/x86_64-unknown-openbsd/libc/)
  * [`x86_64-rumprun-netbsd`](https://doc.rust-lang.org/libc/x86_64-unknown-netbsd/libc/)

The following may be supported, but are not guaranteed to always work:

  * `i686-unknown-freebsd`
  * [`x86_64-unknown-bitrig`](https://doc.rust-lang.org/libc/x86_64-unknown-bitrig/libc/)
  * [`x86_64-unknown-dragonfly`](https://doc.rust-lang.org/libc/x86_64-unknown-dragonfly/libc/)
  * `i686-unknown-haiku`
  * `x86_64-unknown-haiku`
  * [`x86_64-unknown-netbsd`](https://doc.rust-lang.org/libc/x86_64-unknown-netbsd/libc/)
