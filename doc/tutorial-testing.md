% Rust Testing Tutorial

# Quick start

To create test functions, add a `#[test]` attribute like this:

```rust
fn return_two() -> int {
    2
}

#[test]
fn return_two_test() {
    let x = return_two();
    assert!(x == 2);
}
```

To run these tests, use `rustc --test`:

```
$ rustc --test foo.rs; ./foo
running 1 test
test return_two_test ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

`rustc foo.rs` will *not* compile the tests, since `#[test]` implies
`#[cfg(test)]`. The `--test` flag to `rustc` implies `--cfg test`.


# Unit testing in Rust

Rust has built in support for simple unit testing. Functions can be
marked as unit tests using the 'test' attribute.

```rust
#[test]
fn return_none_if_empty() {
    // ... test code ...
}
```

A test function's signature must have no arguments and no return
value. To run the tests in a crate, it must be compiled with the
'--test' flag: `rustc myprogram.rs --test -o myprogram-tests`. Running
the resulting executable will run all the tests in the crate. A test
is considered successful if its function returns; if the task running
the test fails, through a call to `fail!`, a failed `check` or
`assert`, or some other (`assert_eq`, `assert_approx_eq`, ...) means,
then the test fails.

When compiling a crate with the '--test' flag '--cfg test' is also
implied, so that tests can be conditionally compiled.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn return_none_if_empty() {
      // ... test code ...
    }
}
```

Additionally #[test] items behave as if they also have the
#[cfg(test)] attribute, and will not be compiled when the --test flag
is not used.

Tests that should not be run can be annotated with the 'ignore'
attribute. The existence of these tests will be noted in the test
runner output, but the test will not be run. Tests can also be ignored
by configuration so, for example, to ignore a test on windows you can
write `#[ignore(cfg(target_os = "win32"))]`.

Tests that are intended to fail can be annotated with the
'should_fail' attribute. The test will be run, and if it causes its
task to fail then the test will be counted as successful; otherwise it
will be counted as a failure. For example:

```rust
#[test]
#[should_fail]
fn test_out_of_bounds_failure() {
    let v: [int] = [];
    v[0];
}
```

A test runner built with the '--test' flag supports a limited set of
arguments to control which tests are run: the first free argument
passed to a test runner specifies a filter used to narrow down the set
of tests being run; the '--ignored' flag tells the test runner to run
only tests with the 'ignore' attribute.

## Parallelism

By default, tests are run in parallel, which can make interpreting
failure output difficult. In these cases you can set the
`RUST_TEST_TASKS` environment variable to 1 to make the tests run
sequentially.

## Benchmarking

The test runner also understands a simple form of benchmark execution.
Benchmark functions are marked with the `#[bench]` attribute, rather
than `#[test]`, and have a different form and meaning. They are
compiled along with `#[test]` functions when a crate is compiled with
`--test`, but they are not run by default. To run the benchmark
component of your testsuite, pass `--bench` to the compiled test
runner.

The type signature of a benchmark function differs from a unit test:
it takes a mutable reference to type `test::BenchHarness`. Inside the
benchmark function, any time-variable or "setup" code should execute
first, followed by a call to `iter` on the benchmark harness, passing
a closure that contains the portion of the benchmark you wish to
actually measure the per-iteration speed of.

For benchmarks relating to processing/generating data, one can set the
`bytes` field to the number of bytes consumed/produced in each
iteration; this will used to show the throughput of the benchmark.
This must be the amount used in each iteration, *not* the total
amount.

For example:

```rust
extern mod extra;
use std::vec;

#[bench]
fn bench_sum_1024_ints(b: &mut extra::test::BenchHarness) {
    let v = vec::from_fn(1024, |n| n);
    b.iter(|| {v.iter().fold(0, |old, new| old + *new);} );
}

#[bench]
fn initialise_a_vector(b: &mut extra::test::BenchHarness) {
    b.iter(|| {vec::from_elem(1024, 0u64);} );
    b.bytes = 1024 * 8;
}
```

The benchmark runner will calibrate measurement of the benchmark
function to run the `iter` block "enough" times to get a reliable
measure of the per-iteration speed.

Advice on writing benchmarks:

  - Move setup code outside the `iter` loop; only put the part you
    want to measure inside
  - Make the code do "the same thing" on each iteration; do not
    accumulate or change state
  - Make the outer function idempotent too; the benchmark runner is
    likely to run it many times
  - Make the inner `iter` loop short and fast so benchmark runs are
    fast and the calibrator can adjust the run-length at fine
    resolution
  - Make the code in the `iter` loop do something simple, to assist in
    pinpointing performance improvements (or regressions)

To run benchmarks, pass the `--bench` flag to the compiled
test-runner. Benchmarks are compiled-in but not executed by default.

## Examples

### Typical test run

```
> mytests

running 30 tests
running driver::tests::mytest1 ... ok
running driver::tests::mytest2 ... ignored
... snip ...
running driver::tests::mytest30 ... ok

result: ok. 28 passed; 0 failed; 2 ignored
```

### Test run with failures

```
> mytests

running 30 tests
running driver::tests::mytest1 ... ok
running driver::tests::mytest2 ... ignored
... snip ...
running driver::tests::mytest30 ... FAILED

result: FAILED. 27 passed; 1 failed; 2 ignored
```

### Running ignored tests

```
> mytests --ignored

running 2 tests
running driver::tests::mytest2 ... failed
running driver::tests::mytest10 ... ok

result: FAILED. 1 passed; 1 failed; 0 ignored
```

### Running a subset of tests

```
> mytests mytest1

running 11 tests
running driver::tests::mytest1 ... ok
running driver::tests::mytest10 ... ignored
... snip ...
running driver::tests::mytest19 ... ok

result: ok. 11 passed; 0 failed; 1 ignored
```

### Running benchmarks

```
> mytests --bench

running 2 tests
test bench_sum_1024_ints ... bench: 709 ns/iter (+/- 82)
test initialise_a_vector ... bench: 424 ns/iter (+/- 99) = 19320 MB/s

test result: ok. 0 passed; 0 failed; 0 ignored; 2 measured
```

## Saving and ratcheting metrics

When running benchmarks or other tests, the test runner can record
per-test "metrics". Each metric is a scalar `f64` value, plus a noise
value which represents uncertainty in the measurement. By default, all
`#[bench]` benchmarks are recorded as metrics, which can be saved as
JSON in an external file for further reporting.

In addition, the test runner supports _ratcheting_ against a metrics
file. Ratcheting is like saving metrics, except that after each run,
if the output file already exists the results of the current run are
compared against the contents of the existing file, and any regression
_causes the testsuite to fail_. If the comparison passes -- if all
metrics stayed the same (within noise) or improved -- then the metrics
file is overwritten with the new values. In this way, a metrics file
in your workspace can be used to ensure your work does not regress
performance.

Test runners take 3 options that are relevant to metrics:

  - `--save-metrics=<file.json>` will save the metrics from a test run
    to `file.json`
  - `--ratchet-metrics=<file.json>` will ratchet the metrics against
    the `file.json`
  - `--ratchet-noise-percent=N` will override the noise measurements
    in `file.json`, and consider a metric change less than `N%` to be
    noise. This can be helpful if you are testing in a noisy
    environment where the benchmark calibration loop cannot acquire a
    clear enough signal.
