# Running tests

<!-- toc -->

You can run the tests using `x.py`. The most basic command – which
you will almost never want to use! – is as follows:

```bash
./x.py test
```

This will build the stage 1 compiler and then run the whole test
suite. You probably don't want to do this very often, because it takes
a very long time, and anyway bors / GitHub Actions will do it for you.
(Often, I will run this command in the background after opening a PR that
I think is done, but rarely otherwise. -nmatsakis)

The test results are cached and previously successful tests are
`ignored` during testing. The stdout/stderr contents as well as a
timestamp file for every test can be found under `build/ARCH/test/`.
To force-rerun a test (e.g. in case the test runner fails to notice
a change) you can simply remove the timestamp file.

Note that some tests require a Python-enabled gdb. You can test if
your gdb install supports Python by using the `python` command from
within gdb. Once invoked you can type some Python code (e.g.
`print("hi")`) followed by return and then `CTRL+D` to execute it.
If you are building gdb from source, you will need to configure with
`--with-python=<path-to-python-binary>`.

## Running a subset of the test suites

When working on a specific PR, you will usually want to run a smaller
set of tests. For example, a good "smoke test" that can be used after
modifying rustc to see if things are generally working correctly would be the
following:

```bash
./x.py test src/test/ui
```

This will run the `ui` test suite. Of course, the choice
of test suites is somewhat arbitrary, and may not suit the task you are
doing. For example, if you are hacking on debuginfo, you may be better off
with the debuginfo test suite:

```bash
./x.py test src/test/debuginfo
```

If you only need to test a specific subdirectory of tests for any
given test suite, you can pass that directory to `x.py test`:

```bash
./x.py test src/test/ui/const-generics
```

Likewise, you can test a single file by passing its path:

```bash
./x.py test src/test/ui/const-generics/const-test.rs
```

### Run only the tidy script

```bash
./x.py test tidy
```

### Run tests on the standard library

```bash
./x.py test --stage 0 library/std
```

### Run the tidy script and tests on the standard library

```bash
./x.py test --stage 0 tidy library/std
```

### Run tests on the standard library using a stage 1 compiler

```bash
./x.py test library/std
```

By listing which test suites you want to run you avoid having to run
tests for components you did not change at all.

**Warning:** Note that bors only runs the tests with the full stage 2
build; therefore, while the tests **usually** work fine with stage 1,
there are some limitations.

## Running an individual test

Another common thing that people want to do is to run an **individual
test**, often the test they are trying to fix. As mentioned earlier,
you may pass the full file path to achieve this, or alternatively one
may invoke `x.py` with the `--test-args` option:

```bash
./x.py test src/test/ui --test-args issue-1234
```

Under the hood, the test runner invokes the standard rust test runner
(the same one you get with `#[test]`), so this command would wind up
filtering for tests that include "issue-1234" in the name. (Thus
`--test-args` is a good way to run a collection of related tests.)

## Editing and updating the reference files

If you have changed the compiler's output intentionally, or you are
making a new test, you can pass `--bless` to the test subcommand. E.g.
if some tests in `src/test/ui` are failing, you can run

```text
./x.py test src/test/ui --bless
```

to automatically adjust the `.stderr`, `.stdout` or `.fixed` files of
all tests. Of course you can also target just specific tests with the
`--test-args your_test_name` flag, just like when running the tests.

## Passing `--pass $mode`

Pass UI tests now have three modes, `check-pass`, `build-pass` and
`run-pass`. When `--pass $mode` is passed, these tests will be forced
to run under the given `$mode` unless the directive `// ignore-pass`
exists in the test file. For example, you can run all the tests in
`src/test/ui` as `check-pass`:

```bash
./x.py test src/test/ui --pass check
```

By passing `--pass $mode`, you can reduce the testing time. For each
mode, please see [here][mode].

[mode]: ./adding.md#tests-that-do-not-result-in-compile-errors

## Using incremental compilation

You can further enable the `--incremental` flag to save additional
time in subsequent rebuilds:

```bash
./x.py test src/test/ui --incremental --test-args issue-1234
```

If you don't want to include the flag with every command, you can
enable it in the `config.toml`:

```toml
[rust]
incremental = true
```

Note that incremental compilation will use more disk space than usual.
If disk space is a concern for you, you might want to check the size
of the `build` directory from time to time.

## Running tests with different "compare modes"

UI tests may have different output depending on certain "modes" that
the compiler is in. For example, when in "non-lexical lifetimes" (NLL)
mode a test `foo.rs` will first look for expected output in
`foo.nll.stderr`, falling back to the usual `foo.stderr` if not found.
To run the UI test suite in NLL mode, one would use the following:

```bash
./x.py test src/test/ui --compare-mode=nll
```

Other examples of compare-modes are "noopt", "migrate", and
[revisions](./adding.html#revisions).

## Running tests manually

Sometimes it's easier and faster to just run the test by hand. Most tests are
just `rs` files, so you can do something like

```bash
rustc +stage1 src/test/ui/issue-1234.rs
```

This is much faster, but doesn't always work. For example, some tests
include directives that specify specific compiler flags, or which rely
on other crates, and they may not run the same without those options.
