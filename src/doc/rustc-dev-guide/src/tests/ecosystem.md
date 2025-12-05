# Ecosystem testing

Rust tests integration with real-world code in the ecosystem to catch
regressions and make informed decisions about the evolution of the language.

## Testing methods

### Crater

Crater is a tool which runs tests on many thousands of public projects. This
tool has its own separate infrastructure for running, and is not run as part of
CI. See the [Crater chapter](crater.md) for more details.

### `cargotest`

`cargotest` is a small tool which runs `cargo test` on a few sample projects
(such as `servo`, `ripgrep`, `tokei`, etc.). This runs as part of CI and ensures
there aren't any significant regressions:

```console
./x test src/tools/cargotest
```

### Large OSS Project builders

We have CI jobs that build large open-source Rust projects that are used as
regression tests in CI. Our integration jobs build the following projects:

- [Fuchsia](./ecosystem-test-jobs/fuchsia.md)
- [Rust for Linux](./ecosystem-test-jobs/rust-for-linux.md)
