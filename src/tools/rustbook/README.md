# Rustbook

This is a wrapper around [`mdbook`](https://github.com/rust-lang/mdBook/), which is used to generate the book-style documentation in the Rust project. This wrapper serves a few purposes:

- Avoids some of mdbook's large, optional dependencies (like tokio, webserver, etc.).
- Makes it a little easier to customize and override some of mdbook's behaviors (like swapping in custom preprocessors).
- Supports vendoring of the source via Rust's normal release process.

This is invoked automatically when building mdbook-style documentation, for example via `./x doc`.

## Cargo workspace

This package defines a separate cargo workspace from the main Rust workspace for a few reasons (ref [#127786](https://github.com/rust-lang/rust/pull/127786):

- Avoids requiring checking out submodules for developers who are not working on the documentation. Otherwise, some submodules such as those that have custom preprocessors would be required for cargo to find the dependencies.
- Avoids problems with updating dependencies. Unfortunately this workspace has a rather large set of dependencies, which can make coordinating updates difficult (see [#127890](https://github.com/rust-lang/rust/issues/127890)).

## Custom preprocessors

Some books have custom mdbook preprocessors that need to be integrated with both the book's repository, and the build system here in the `rust-lang/rust` repository. To add a new preprocessor, there are few things to do:

1. Implement the preprocessor as a cargo library in the book's repository.
2. Add the `[preprocessor]` table to the book's `book.toml` file. I recommend setting the command so that the preprocessor gets built automatically. It may look something like:
  ```toml
  [preprocessor.spec]
  command = "cargo run --manifest-path my-cool-extension/Cargo.toml"

  [build]
  extra-watch-dirs = ["my-cool-extension/src"]
  ```
3. Add the preprocessor as a dependency in rustbook's `Cargo.toml`.
4. Call `with_preprocessor` in `rustbook/src/main.rs`.
5. Be sure to test that it generates correctly, such as running `./x doc MY-BOOK-NAME --open` and verify the content looks correct.
6. Also test tidy and your book, such as running `./x test tidy` and `./x test MY-BOOK-NAME`.
