# Benchmarking Clippy

Benchmarking Clippy is similar to using our Lintcheck tool, in fact, it even
uses the same tool! Just by adding a `--perf` flag it will transform Lintcheck
into a very simple but powerful benchmarking tool!

It requires having the [`perf` tool][perf] installed, as `perf` is what's actually
profiling Clippy under the hood.

The lintcheck `--perf` tool generates a series of `perf.data` in the
`target/lintcheck/sources/<package>-<version>` directories. Each `perf.data`
corresponds to the package which is contained.

Lintcheck uses the `-g` flag, meaning that you can get stack traces for richer
analysis, including with tools such as [flamegraph][flamegraph-perf]
(or [`flamegraph-rs`][flamegraph-rs]).

Currently, we only measure instruction count, as it's the most reproducible metric
and [rustc-perf][rustc-perf] also considers it the main number to focus on.

## Benchmarking a PR

Having a benchmarking tool directly implemented into lintcheck gives us the
ability to benchmark any given PR just by making a before and after 

Here's the way you can get into any PR, benchmark it, and then benchmark
`master`.

The first `perf.data` will not have any numbers appended, but any subsequent
benchmark will be written to `perf.data.number` with a number growing for 0.
All benchmarks are compressed so that you can

```bash
git fetch upstream pull/<PR_NUMBER>/head:<BRANCH_NAME>
git switch BRANCHNAME

# Bench
cargo lintcheck --perf

# Get last common commit, checkout that
LAST_COMMIT=$(git log BRANCHNAME..master --oneline | tail -1 | cut -c 1-11)
git switch -c temporary $LAST_COMMIT

# We're now on master

# Bench
cargo lintcheck --perf
perf diff ./target/lintcheck/sources/CRATE/perf.data ./target/lintcheck/sources/CRATE/perf.data.0
```


[perf]: https://perfwiki.github.io/main/
[flamegraph-perf]: https://github.com/brendangregg/FlameGraph
[flamegraph-rs]: https://github.com/flamegraph-rs/flamegraph
[rustc-perf]: https://github.com/rust-lang/rustc-perf
