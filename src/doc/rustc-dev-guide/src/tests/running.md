# Running tests

<!-- toc -->

You can run the entire test collection using `x`. But note that running the
*entire* test collection is almost never what you want to do during local
development because it takes a really long time. For local development, see the
subsection after on how to run a subset of tests.

<div class="warning">

Running plain `./x test` will build the stage 1 compiler and then run the whole
test suite. This not only include `tests/`, but also `library/`, `compiler/`,
`src/tools/` package tests and more.

You usually only want to run a subset of the test suites (or even a smaller set
of tests than that) which you expect will exercise your changes. PR CI exercises
a subset of test collections, and merge queue CI will exercise all of the test
collection.

</div>

```text
./x test
```

The test results are cached and previously successful tests are `ignored` during
testing. The stdout/stderr contents as well as a timestamp file for every test
can be found under `build/<target-tuple>/test/` for the given
`<target-tuple>`. To force-rerun a test (e.g. in case the test runner fails to
notice a change) you can use the `--force-rerun` CLI option.

> **Note on requirements of external dependencies**
>
> Some test suites may require external dependencies. This is especially true of
> debuginfo tests. Some debuginfo tests require a Python-enabled gdb. You can
> test if your gdb install supports Python by using the `python` command from
> within gdb. Once invoked you can type some Python code (e.g. `print("hi")`)
> followed by return and then `CTRL+D` to execute it. If you are building gdb
> from source, you will need to configure with
> `--with-python=<path-to-python-binary>`.

## Running a subset of the test suites

When working on a specific PR, you will usually want to run a smaller set of
tests. For example, a good "smoke test" that can be used after modifying rustc
to see if things are generally working correctly would be to exercise the `ui`
test suite ([`tests/ui`]):

```text
./x test tests/ui
```

Of course, the choice of test suites is
somewhat arbitrary, and may not suit the task you are doing. For example, if you
are hacking on debuginfo, you may be better off with the debuginfo test suite:

```text
./x test tests/debuginfo
```

If you only need to test a specific subdirectory of tests for any given test
suite, you can pass that directory as a filter to `./x test`:

```text
./x test tests/ui/const-generics
```

> **Note for MSYS2**
>
> On MSYS2 the paths seem to be strange and `./x test` neither recognizes
> `tests/ui/const-generics` nor `tests\ui\const-generics`. In that case, you can
> workaround it by using e.g. `./x test ui
> --test-args="tests/ui/const-generics"`.

Likewise, you can test a single file by passing its path:

```text
./x test tests/ui/const-generics/const-test.rs
```

`x` doesn't support running a single tool test by passing its path yet. You'll
have to use the `--test-args` argument as described
[below](#running-an-individual-test).

```text
./x test src/tools/miri --test-args tests/fail/uninit/padding-enum.rs
```

### Run only the tidy script

```text
./x test tidy
```

### Run tests on the standard library

```text
./x test --stage 0 library/std
```

Note that this only runs tests on `std`; if you want to test `core` or other
crates, you have to specify those explicitly.

### Run the tidy script and tests on the standard library

```text
./x test --stage 0 tidy library/std
```

### Run tests on the standard library using a stage 1 compiler

```text
./x test --stage 1 library/std
```

By listing which test suites you want to run,
you avoid having to run tests for components you did not change at all.

<div class="warning">

Note that bors only runs the tests with the full stage 2 build; therefore, while
the tests **usually** work fine with stage 1, there are some limitations.

</div>

### Run all tests using a stage 2 compiler

```text
./x test --stage 2
```

<div class="warning">
You almost never need to do this; CI will run these tests for you.
</div>

## Run unit tests on the compiler/library

You may want to run unit tests on a specific file with following:

```text
./x test compiler/rustc_data_structures/src/thin_vec/tests.rs
```

But unfortunately, it's impossible. You should invoke the following instead:

```text
./x test compiler/rustc_data_structures/ --test-args thin_vec
```

## Running an individual test

Another common thing that people want to do is to run an **individual test**,
often the test they are trying to fix. As mentioned earlier, you may pass the
full file path to achieve this, or alternatively one may invoke `x` with the
`--test-args` option:

```text
./x test tests/ui --test-args issue-1234
```

Under the hood, the test runner invokes the standard Rust test runner (the same
one you get with `#[test]`), so this command would wind up filtering for tests
that include "issue-1234" in the name. Thus, `--test-args` is a good way to run
a collection of related tests.

## Passing arguments to `rustc` when running tests

It can sometimes be useful to run some tests with specific compiler arguments,
without using `RUSTFLAGS` (during development of unstable features, with `-Z`
flags, for example).

This can be done with `./x test`'s `--compiletest-rustc-args` option, to pass
additional arguments to the compiler when building the tests.

## Editing and updating the reference files

If you have changed the compiler's output intentionally, or you are making a new
test, you can pass `--bless` to the test subcommand.

As an example,
if some tests in `tests/ui` are failing, you can run this command:

```text
./x test tests/ui --bless
```

It automatically adjusts the `.stderr`, `.stdout`, or `.fixed` files of all `test/ui` tests.
Of course you can also target just specific tests with the `--test-args your_test_name` flag,
just like when running the tests without the `--bless` flag.

## Configuring test running

There are a few options for running tests:

* `bootstrap.toml` has the `rust.verbose-tests` option. If `false`, each test will
  print a single dot (the default). If `true`, the name of every test will be
  printed. This is equivalent to the `--quiet` option in the [Rust test
  harness](https://doc.rust-lang.org/rustc/tests/).
* The environment variable `RUST_TEST_THREADS` can be set to the number of
  concurrent threads to use for testing.

## Passing `--pass $mode`

Pass UI tests now have three modes, `check-pass`, `build-pass` and `run-pass`.
When `--pass $mode` is passed, these tests will be forced to run under the given
`$mode` unless the directive `//@ ignore-pass` exists in the test file. For
example, you can run all the tests in `tests/ui` as `check-pass`:

```text
./x test tests/ui --pass check
```

By passing `--pass $mode`, you can reduce the testing time. For each mode,
please see [Controlling pass/fail
expectations](ui.md#controlling-passfail-expectations).

## Running tests with different "compare modes"

UI tests may have different output depending on certain "modes" that the
compiler is in. For example, when using the Polonius mode, a test `foo.rs` will
first look for expected output in `foo.polonius.stderr`, falling back to the
usual `foo.stderr` if not found. The following will run the UI test suite in
Polonius mode:

```text
./x test tests/ui --compare-mode=polonius
```

See [Compare modes](compiletest.md#compare-modes) for more details.

## Running tests manually

Sometimes it's easier and faster to just run the test by hand. Most tests are
just `.rs` files, so after [creating a rustup
toolchain](../building/how-to-build-and-run.md#creating-a-rustup-toolchain), you
can do something like:

```text
rustc +stage1 tests/ui/issue-1234.rs
```

This is much faster, but doesn't always work. For example, some tests include
directives that specify specific compiler flags, or which rely on other crates,
and they may not run the same without those options.

## Running tests on a remote machine

Tests may be run on a remote machine (e.g. to test builds for a different
architecture). This is done using `remote-test-client` on the build machine to
send test programs to `remote-test-server` running on the remote machine.
`remote-test-server` executes the test programs and sends the results back to
the build machine. `remote-test-server` provides *unauthenticated remote code
execution* so be careful where it is used.

To do this, first build `remote-test-server` for the remote machine, e.g. for
RISC-V

```text
./x build src/tools/remote-test-server --target riscv64gc-unknown-linux-gnu
```

The binary will be created at
`./build/host/stage2-tools/$TARGET_ARCH/release/remote-test-server`. Copy this
over to the remote machine.

On the remote machine, run the `remote-test-server` with the `--bind
0.0.0.0:12345` flag (and optionally `-v` for verbose output). Output should look
like this:

```text
$ ./remote-test-server -v --bind 0.0.0.0:12345
starting test server
listening on 0.0.0.0:12345!
```

Note that binding the server to 0.0.0.0 will allow all hosts able to reach your
machine to execute arbitrary code on your machine. We strongly recommend either
setting up a firewall to block external access to port 12345, or to use a more
restrictive IP address when binding.

You can test if the `remote-test-server` is working by connecting to it and
sending `ping\n`. It should reply `pong`:

```text
$ nc $REMOTE_IP 12345
ping
pong
```

To run tests using the remote runner, set the `TEST_DEVICE_ADDR` environment
variable then use `x` as usual. For example, to run `ui` tests for a RISC-V
machine with the IP address `1.2.3.4` use

```text
export TEST_DEVICE_ADDR="1.2.3.4:12345"
./x test tests/ui --target riscv64gc-unknown-linux-gnu
```

If `remote-test-server` was run with the verbose flag, output on the test
machine may look something like

```text
[...]
run "/tmp/work/test1007/a"
run "/tmp/work/test1008/a"
run "/tmp/work/test1009/a"
run "/tmp/work/test1010/a"
run "/tmp/work/test1011/a"
run "/tmp/work/test1012/a"
run "/tmp/work/test1013/a"
run "/tmp/work/test1014/a"
run "/tmp/work/test1015/a"
run "/tmp/work/test1016/a"
run "/tmp/work/test1017/a"
run "/tmp/work/test1018/a"
[...]
```

Tests are built on the machine running `x` not on the remote machine. Tests
which fail to build unexpectedly (or `ui` tests producing incorrect build
output) may fail without ever running on the remote machine.

## Testing on emulators

Some platforms are tested via an emulator for architectures that aren't readily
available. For architectures where the standard library is well supported and
the host operating system supports TCP/IP networking, see the above instructions
for testing on a remote machine (in this case the remote machine is emulated).

There is also a set of tools for orchestrating running the tests within the
emulator. Platforms such as `arm-android` and `arm-unknown-linux-gnueabihf` are
set up to automatically run the tests under emulation on GitHub Actions. The
following will take a look at how a target's tests are run under emulation.

The Docker image for [armhf-gnu] includes [QEMU] to emulate the ARM CPU
architecture. Included in the Rust tree are the tools [remote-test-client] and
[remote-test-server] which are programs for sending test programs and libraries
to the emulator, and running the tests within the emulator, and reading the
results.  The Docker image is set up to launch `remote-test-server` and the
build tools use `remote-test-client` to communicate with the server to
coordinate running tests (see [src/bootstrap/src/core/build_steps/test.rs]).

> **TODO**
>
> - Is there any support for using an iOS emulator?
> - It's also unclear to me how the wasm or asm.js tests are run.

[armhf-gnu]: https://github.com/rust-lang/rust/tree/master/src/ci/docker/host-x86_64/armhf-gnu/Dockerfile
[QEMU]: https://www.qemu.org/
[remote-test-client]: https://github.com/rust-lang/rust/tree/master/src/tools/remote-test-client
[remote-test-server]: https://github.com/rust-lang/rust/tree/master/src/tools/remote-test-server
[src/bootstrap/src/core/build_steps/test.rs]: https://github.com/rust-lang/rust/blob/master/src/bootstrap/src/core/build_steps/test.rs

## Running rustc_codegen_gcc tests

First thing to know is that it only supports linux x86_64 at the moment. We will
extend its support later on.

You need to update `codegen-backends` value in your `bootstrap.toml` file in the
`[rust]` section and add "gcc" in the array:

```toml
codegen-backends = ["llvm", "gcc"]
```

Then you need to install libgccjit 12. For example with `apt`:

```text
apt install libgccjit-12-dev
```

Now you can run the following command:

```text
./x test compiler/rustc_codegen_gcc/
```

If it cannot find the `.so` library (if you installed it with `apt` for example), you
need to pass the library file path with `LIBRARY_PATH`:

```text
LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/12/ ./x test compiler/rustc_codegen_gcc/
```

If you encounter bugs or problems, don't hesitate to open issues on the
[`rustc_codegen_gcc`
repository](https://github.com/rust-lang/rustc_codegen_gcc/).

[`tests/ui`]: https://github.com/rust-lang/rust/tree/master/tests/ui
