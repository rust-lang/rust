# Travis CI

You can add Clippy to Travis CI in the same way you use it locally:

```yml
language: rust
rust:
  - stable
  - beta
before_script:
  - rustup component add clippy
script:
  - cargo clippy
  # if you want the build job to fail when encountering warnings, use
  - cargo clippy -- -D warnings
  # in order to also check tests and non-default crate features, use
  - cargo clippy --all-targets --all-features -- -D warnings
  - cargo test
  # etc.
```

Note that adding `-D warnings` will cause your build to fail if **any** warnings are found in your code.
That includes warnings found by rustc (e.g. `dead_code`, etc.). If you want to avoid this and only cause
an error for Clippy warnings, use `#![deny(clippy::all)]` in your code or `-D clippy::all` on the command
line. (You can swap `clippy::all` with the specific lint category you are targeting.)
