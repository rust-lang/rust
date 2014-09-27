% The Rust Testing Guide

# Quick start

To create test functions, add a `#[test]` attribute like this:

~~~test_harness
fn return_two() -> int {
    2
}

#[test]
fn return_two_test() {
    let x = return_two();
    assert!(x == 2);
}
~~~

To run these tests, compile with `rustc --test` and run the resulting
binary:

~~~console
$ rustc --test foo.rs
$ ./foo
running 1 test
test return_two_test ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
~~~

`rustc foo.rs` will *not* compile the tests, since `#[test]` implies
`#[cfg(test)]`. The `--test` flag to `rustc` implies `--cfg test`.


# Unit testing in Rust

Rust has built in support for simple unit testing. Functions can be
marked as unit tests using the `test` attribute.

~~~test_harness
#[test]
fn return_none_if_empty() {
    // ... test code ...
}
~~~

A test function's signature must have no arguments and no return
value. To run the tests in a crate, it must be compiled with the
`--test` flag: `rustc myprogram.rs --test -o myprogram-tests`. Running
the resulting executable will run all the tests in the crate. A test
is considered successful if its function returns; if the task running
the test fails, through a call to `fail!`, a failed `assert`, or some
other (`assert_eq`, ...) means, then the test fails.

When compiling a crate with the `--test` flag `--cfg test` is also
implied, so that tests can be conditionally compiled.

~~~test_harness
#[cfg(test)]
mod tests {
    #[test]
    fn return_none_if_empty() {
      // ... test code ...
    }
}
~~~

Additionally `#[test]` items behave as if they also have the
`#[cfg(test)]` attribute, and will not be compiled when the `--test` flag
is not used.

Tests that should not be run can be annotated with the `ignore`
attribute. The existence of these tests will be noted in the test
runner output, but the test will not be run. Tests can also be ignored
by configuration using the `cfg_attr` attribute so, for example, to ignore a
test on windows you can write `#[cfg_attr(windows, ignore)]`.

Tests that are intended to fail can be annotated with the
`should_fail` attribute. The test will be run, and if it causes its
task to fail then the test will be counted as successful; otherwise it
will be counted as a failure. For example:

~~~test_harness
#[test]
#[should_fail]
fn test_out_of_bounds_failure() {
    let v: &[int] = [];
    v[0];
}
~~~

A test runner built with the `--test` flag supports a limited set of
arguments to control which tests are run:

- the first free argument passed to a test runner is interpreted as a
  regular expression
  ([syntax reference](regex/index.html#syntax))
  and is used to narrow down the set of tests being run. Note: a plain
  string is a valid regular expression that matches itself.
- the `--ignored` flag tells the test runner to run only tests with the
  `ignore` attribute.

## Parallelism

By default, tests are run in parallel, which can make interpreting
failure output difficult. In these cases you can set the
`RUST_TEST_TASKS` environment variable to 1 to make the tests run
sequentially.

## Examples

### Typical test run

~~~console
$ mytests

running 30 tests
running driver::tests::mytest1 ... ok
running driver::tests::mytest2 ... ignored
... snip ...
running driver::tests::mytest30 ... ok

result: ok. 28 passed; 0 failed; 2 ignored
~~~

### Test run with failures

~~~console
$ mytests

running 30 tests
running driver::tests::mytest1 ... ok
running driver::tests::mytest2 ... ignored
... snip ...
running driver::tests::mytest30 ... FAILED

result: FAILED. 27 passed; 1 failed; 2 ignored
~~~

### Running ignored tests

~~~console
$ mytests --ignored

running 2 tests
running driver::tests::mytest2 ... failed
running driver::tests::mytest10 ... ok

result: FAILED. 1 passed; 1 failed; 0 ignored
~~~

### Running a subset of tests

Using a plain string:

~~~console
$ mytests mytest23

running 1 tests
running driver::tests::mytest23 ... ok

result: ok. 1 passed; 0 failed; 0 ignored
~~~

Using some regular expression features:

~~~console
$ mytests 'mytest[145]'

running 13 tests
running driver::tests::mytest1 ... ok
running driver::tests::mytest4 ... ok
running driver::tests::mytest5 ... ok
running driver::tests::mytest10 ... ignored
... snip ...
running driver::tests::mytest19 ... ok

result: ok. 13 passed; 0 failed; 1 ignored
~~~

# Microbenchmarking

The test runner also understands a simple form of benchmark execution.
Benchmark functions are marked with the `#[bench]` attribute, rather
than `#[test]`, and have a different form and meaning. They are
compiled along with `#[test]` functions when a crate is compiled with
`--test`, but they are not run by default. To run the benchmark
component of your testsuite, pass `--bench` to the compiled test
runner.

The type signature of a benchmark function differs from a unit test:
it takes a mutable reference to type
`test::Bencher`. Inside the benchmark function, any
time-variable or "setup" code should execute first, followed by a call
to `iter` on the benchmark harness, passing a closure that contains
the portion of the benchmark you wish to actually measure the
per-iteration speed of.

For benchmarks relating to processing/generating data, one can set the
`bytes` field to the number of bytes consumed/produced in each
iteration; this will be used to show the throughput of the benchmark.
This must be the amount used in each iteration, *not* the total
amount.

For example:

~~~test_harness
extern crate test;

use test::Bencher;

#[bench]
fn bench_sum_1024_ints(b: &mut Bencher) {
    let v = Vec::from_fn(1024, |n| n);
    b.iter(|| v.iter().fold(0, |old, new| old + *new));
}

#[bench]
fn initialise_a_vector(b: &mut Bencher) {
    b.iter(|| Vec::from_elem(1024, 0u64));
    b.bytes = 1024 * 8;
}
~~~

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

~~~console
$ rustc mytests.rs -O --test
$ mytests --bench

running 2 tests
test bench_sum_1024_ints ... bench: 709 ns/iter (+/- 82)
test initialise_a_vector ... bench: 424 ns/iter (+/- 99) = 19320 MB/s

test result: ok. 0 passed; 0 failed; 0 ignored; 2 measured
~~~

## Benchmarks and the optimizer

Benchmarks compiled with optimizations activated can be dramatically
changed by the optimizer so that the benchmark is no longer
benchmarking what one expects. For example, the compiler might
recognize that some calculation has no external effects and remove
it entirely.

~~~test_harness
extern crate test;
use test::Bencher;

#[bench]
fn bench_xor_1000_ints(b: &mut Bencher) {
    b.iter(|| {
        range(0u, 1000).fold(0, |old, new| old ^ new);
    });
}
~~~

gives the following results

~~~console
running 1 test
test bench_xor_1000_ints ... bench:         0 ns/iter (+/- 0)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured
~~~

The benchmarking runner offers two ways to avoid this. Either, the
closure that the `iter` method receives can return an arbitrary value
which forces the optimizer to consider the result used and ensures it
cannot remove the computation entirely. This could be done for the
example above by adjusting the `bh.iter` call to

~~~
# struct X; impl X { fn iter<T>(&self, _: || -> T) {} } let b = X;
b.iter(|| {
    // note lack of `;` (could also use an explicit `return`).
    range(0u, 1000).fold(0, |old, new| old ^ new)
});
~~~

Or, the other option is to call the generic `test::black_box`
function, which is an opaque "black box" to the optimizer and so
forces it to consider any argument as used.

~~~
extern crate test;

# fn main() {
# struct X; impl X { fn iter<T>(&self, _: || -> T) {} } let b = X;
b.iter(|| {
    test::black_box(range(0u, 1000).fold(0, |old, new| old ^ new));
});
# }
~~~

Neither of these read or modify the value, and are very cheap for
small values. Larger values can be passed indirectly to reduce
overhead (e.g. `black_box(&huge_struct)`).

Performing either of the above changes gives the following
benchmarking results

~~~console
running 1 test
test bench_xor_1000_ints ... bench:       375 ns/iter (+/- 148)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured
~~~

However, the optimizer can still modify a testcase in an undesirable
manner even when using either of the above. Benchmarks can be checked
by hand by looking at the output of the compiler using the `--emit=ir`
(for LLVM IR), `--emit=asm` (for assembly) or compiling normally and
using any method for examining object code.

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
