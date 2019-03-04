WIP - stable libtest
===

The migration of libtest to stable Rust is currently in progress.

You can find libtest at: https://github.com/rust-lang/libtest . If you need to
make a change:

* perform the change there, 
* do a new crates.io release, and
* send a PR to rust-lang/rust bumping the libtest version.

## Roadmap

Right now all the contests of libtest live in the external repo. 

The next steps are:

* make `#[test]` and `#[ignore]` procedural macros in the prelude by default,
  routed to the same procedural macro, so that it doesn't matter which one runs
  first.
* move the unstable APIs back into rust-lang/rust to help maintainability
  (replacing `pub use libtest::*` with explicit imports)
* migrate libtest to the real `term` crate
* provide `libtest` a real `custom_test_framework` runner (in parallel with the
  runner in rust-lang/rust)
* set up `libtest` as a normal `custom_test_framework` inside rust-lang/rust
* refactor the internal structure of `libtest` to make it re-usable by
  third-party custom test frameworks (think test formatting, benchmark
  formatting, argument parsing, json format serialization, etc.)
