# Integration testing

Rust tests integration with real-world code to catch regressions and make
informed decisions about the evolution of the language.

## Testing methods

### Crater

Crater is a tool which runs tests on many thousands of public projects. This
tool has its own separate infrastructure for running, and is not run as part of
CI. See the [Crater chapter](crater.md) for more details.

### Cargo test

`cargotest` is a small tool which runs `cargo test` on a few sample projects
(such as `servo`, `ripgrep`, `tokei`, etc.).
This runs as part of CI and ensures there aren't any significant regressions.

> Example: `./x test src/tools/cargotest`

### Integration builders

Integration jobs build large open-source Rust projects that are used as
regression tests in CI. Our integration jobs build the following projects:

- [Fuchsia](fuchsia.md)
- [Rust for Linux](rust-for-linux.md)

## A note about terminology

The term "integration testing" can be used to mean many things. Many of the
compiletest tests within the Rust repo could be justifiably called integration
tests, because they test the integration of many parts of the compiler, or test
the integration of the compiler with other external tools. Calling all of them
integration tests would not be very helpful, especially since those kinds of
tests already have their own specialized names.

We use the term "integration" here to mean integrating the Rust compiler and
toolchain with the ecosystem of Rust projects that depend on it. This is partly
for lack of a better term, but it also reflects a difference in testing approach
from other projects and the comparative advantage it implies.

The Rust compiler is part of the ecosystem, and the ecosystem is in many cases
part of Rust, both in terms of libraries it uses and in terms of the efforts of many
contributors who come to "scratch their own itch". Finally, because Rust has the
ability to do integration testing at such a broad scale, it shortens development
cycles by finding defects earlier.

