# Running tests

You can run the tests using `x.py`. The most basic command – which
you will almost never want to use! – is as follows:

```bash
> ./x.py test
```

This will build the full stage 2 compiler and then run the whole test
suite. You probably don't want to do this very often, because it takes
a very long time, and anyway bors / travis will do it for you. (Often,
I will run this command in the background after opening a PR that I
think is done, but rarely otherwise. -nmatsakis)

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
set of tests, and with a stage 1 build. For example, a good "smoke
test" that can be used after modifying rustc to see if things are
generally working correctly would be the following:

```bash
> ./x.py test --stage 1 src/test/{ui,compile-fail,run-pass}
```

This will run the `ui`, `compile-fail`, and `run-pass` test suites,
and only with the stage 1 build. Of course, the choice of test suites
is somewhat arbitrary, and may not suit the task you are doing. For
example, if you are hacking on debuginfo, you may be better off with
the debuginfo test suite:

```bash
> ./x.py test --stage 1 src/test/debuginfo
```

If you only need to test a specific subdirectory of tests for any
given test suite, you can pass that directory to `x.py test`:

```bash
> ./x.py test --stage 1 src/test/ui/const-generics
```

Likewise, you can test a single file by passing its path:

```bash
> ./x.py test --stage 1 src/test/ui/const-generics/const-test.rs
```

### Run only the tidy script

```bash
> ./x.py test src/tools/tidy
```

### Run tests on the standard library

```bash
> ./x.py test src/libstd
```

### Run tests on the standard library and run the tidy script

```bash
> ./x.py test src/libstd src/tools/tidy
```

### Run tests on the standard library using a stage 1 compiler

```bash
>   ./x.py test src/libstd --stage 1
```

By listing which test suites you want to run you avoid having to run
tests for components you did not change at all.

**Warning:** Note that bors only runs the tests with the full stage 2
build; therefore, while the tests **usually** work fine with stage 1,
there are some limitations. In particular, the stage1 compiler doesn't
work well with procedural macros or custom derive tests.

## Running an individual test

Another common thing that people want to do is to run an **individual
test**, often the test they are trying to fix. As mentioned earlier,
you may pass the full file path to achieve this, or alternatively one
may invoke `x.py` with the `--test-args` option:

```bash
> ./x.py test --stage 1 src/test/ui --test-args issue-1234
```

Under the hood, the test runner invokes the standard rust test runner
(the same one you get with `#[test]`), so this command would wind up
filtering for tests that include "issue-1234" in the name. (Thus
`--test-args` is a good way to run a collection of related tests.)

## Using incremental compilation

You can further enable the `--incremental` flag to save additional
time in subsequent rebuilds:

```bash
> ./x.py test --stage 1 src/test/ui --incremental --test-args issue-1234
```

If you don't want to include the flag with every command, you can
enable it in the `config.toml`, too:

```toml
# Whether to always use incremental compilation when building rustc
incremental = true
```

Note that incremental compilation will use more disk space than usual.
If disk space is a concern for you, you might want to check the size
of the `build` directory from time to time.

## Running tests manually

Sometimes it's easier and faster to just run the test by hand. Most tests are
just `rs` files, so you can do something like

```bash
> rustc +stage1 src/test/ui/issue-1234.rs
```

This is much faster, but doesn't always work. For example, some tests
include directives that specify specific compiler flags, or which rely
on other crates, and they may not run the same without those options.
