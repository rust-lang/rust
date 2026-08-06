# Parallel frontend testing on CI

If you see any test failures in `tests/ui` from the CI job `x86_64-gnu-parallel-frontend`, please
add `//@ ignore-parallel-frontend triage` to the failing test, even if your PR is otherwise
entirely unrelated to parallel compiler or its testing.
In some time people from the parallel rustc working group will triage the failing test, make a
tracking issue for it, and try to debug the problem.

For more context, see:

* [MCP: Stabilization strategy for rustc parallel frontend](https://github.com/rust-lang/compiler-team/issues/1005)
* Tracking issue: <https://github.com/rust-lang/rust/issues/118698>
