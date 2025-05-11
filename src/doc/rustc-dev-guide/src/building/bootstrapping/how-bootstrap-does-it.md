# How Bootstrap does it

The core concept in Bootstrap is a build [`Step`],  which are chained together
by [`Builder::ensure`]. [`Builder::ensure`] takes a [`Step`] as input, and runs
the [`Step`] if and only if it has not already been run. Let's take a closer
look at [`Step`].

## Synopsis of [`Step`]

A [`Step`] represents a granular collection of actions involved in the process
of producing some artifact. It can be thought of like a rule in Makefiles.
The [`Step`] trait is defined as:

```rs,no_run
pub trait Step: 'static + Clone + Debug + PartialEq + Eq + Hash {
    type Output: Clone;

    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = false;

    // Required methods
    fn run(self, builder: &Builder<'_>) -> Self::Output;
    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_>;

    // Provided method
    fn make_run(_run: RunConfig<'_>) { ... }
}
```

- `run` is the function that is responsible for doing the work.
  [`Builder::ensure`] invokes `run`.
- `should_run` is the command-line interface, which determines if an invocation
  such as `x build foo` should run a given [`Step`]. In a "default" context
  where no paths are provided, then `make_run` is called directly.
- `make_run` is invoked only for things directly asked via the CLI and not
  for steps which are dependencies of other steps.

## The entry points

There's a couple of preliminary steps before core Bootstrap code is reached:

1. Shell script or `make`: [`./x`](https://github.com/rust-lang/rust/blob/master/x) or [`./x.ps1`](https://github.com/rust-lang/rust/blob/master/x.ps1) or `make`
2. Convenience wrapper script: [`x.py`](https://github.com/rust-lang/rust/blob/master/x.py)
3. [`src/bootstrap/bootstrap.py`](https://github.com/rust-lang/rust/blob/master/src/bootstrap/bootstrap.py)
4. [`src/bootstrap/src/bin/main.rs`](https://github.com/rust-lang/rust/blob/master/src/bootstrap/src/bin/main.rs)

See [src/bootstrap/README.md](https://github.com/rust-lang/rust/blob/master/src/bootstrap/README.md)
for a more specific description of the implementation details.

[`Step`]: https://doc.rust-lang.org/nightly/nightly-rustc/bootstrap/core/builder/trait.Step.html
[`Builder::ensure`]: https://doc.rust-lang.org/nightly/nightly-rustc/bootstrap/core/builder/struct.Builder.html#method.ensure
