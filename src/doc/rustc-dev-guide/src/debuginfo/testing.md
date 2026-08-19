# Testing

> [!IMPORTANT]
> The debug info test suite is undergoing a substantial rewrite. This section will be filled out as
> the rewrite makes progress.
>
> Please see [this tracking issue][148483] for more information.

[148483]: https://github.com/rust-lang/rust/issues/148483

Debug info tests check a few important things:

* Are we outputting the information in the way we expect?
* Is what we output readable by the debugger?
* Do our visualizers work the way we expect?

The first question is typically answered by `tests/codegen-llvm`, but debug info generation is often
tested incidentally, rather than deliberately.
As of <!-- date-check --> Jul 2026, there is a much larger focus on the latter two questions,
and those will be covered in detail here.
The tests that answer those questions live in `tests/debuginfo`, which is executed by `compiletest`.

For much of the test suite's lifespan, debuggers were discovered automatically, and tests were
tests were comprised of `$DEBUGGER-command` and `$DEBUGGER-check` directives (i.e. raw string
comparisons) that checked variable printing, breakpoint locations, etc. Put bluntly, this system was
a nightmare and lead to a [litany of issues](https://github.com/rust-lang/rust/issues/134682).
To help remedy this:

1. [`tests/debuginfo` is now opt-in](https://github.com/rust-lang/rust/pull/159455) for GDB and LLDB.
2. a new directive was added: `$DEBUGGER-repr`.
   This directive dispatches to custom logic that polls
   the debugger for additional information that isn't visible in the printed output.
   It also automatically separates output by target,
   allowing the tests to be run on different platforms without conflicts.

# The `repr` directive

> [!IMPORTANT]
> As of July 2026, this command is only supported by LLDB. GDB support is planned, but
> has not been implemented. It is unclear whether this directive will ever be suited for use with
> CDB.

In short, `$DEBUGGER-repr` commands are desugared to:

```
//@ $DEBUGGER-command:repr $VAR_NAME
//@ $DEBUGGER-check:$VAR_NAME ok
```

When the commands are passed to the debugger, our test framework intercepts `repr` pseudo-commands
and runs special logic on them, testing against data stored in
`tests/debuginfo/<test_name>/input/<debugger>_input/<target_group>.json`.

"Target groups" cover the set of targets where we cannot guarantee identical output.
Those targets are defined by the
 [`Target` enum in `common.py`](https://github.com/rust-lang/rust/blob/bf9944f0b8006b152ef4d5f408ae75a0dde3d044/src/etc/lldb_batchmode/common.py#L54).
As of <!-- date-check --> Jul 2026, this list includes `non_windows`, `windows_gnu`, and `windows_msvc`.
It is intentionally kept as short as possible,
since each target is a new set of test data that must be updated when changes are made.
There is still not a perfect solution for how tests can be
`--bless`-ed by contributors who do not have access to all of the targets.

The input data can be automatically updated for expected changes by adding `--bless` to the test
invocation (e.g. `./x test tests/debuginfo/basic-types/main.rs --bless`).
`--bless` updates the in-memory representation, tests against it,
and if no errors occur, saves the data back to the target file (or creates a new file if necessary).

The schema of the input data is defined by the classes in
[`common.py`](https://github.com/rust-lang/rust/blob/be3d26db984c6f96335faca1f254dc04873cb1c1/src/etc/lldb_batchmode/common.py).
The top-level container is `TargetData`.
This schema is identical for all debuggers.

## Converting existing tests

Nearly any time a variable is tested, the `repr` directive should be preferred over `command` +
`check`.
As of <!-- date-check --> Jul 2026, only a single test has been converted over, but more will follow as
part of the test rewrite mentioned above.
Thankfully, the conversion process is fairly easy.
For a given check:

```
//@ lldb-command:v foo
//@ lldb-check:<foo output>
```

The equivalent `repr` test is:

```
//@ lldb-repr:foo
```

Once all `command` + `checks` are converted to `repr`, run the tests with the `--bless` option.
If you have access to additional targets, `--bless` the data for the remainder of the target groups
as well (e.g. if you are on a Windows machine, bless once for `x86_64-pc-windows-msvc`, once for
`x86_64-pc-windows-gnu`, and use WSL to bless for `x86_64-unknown-linux-gnu`).


## Implementation

### Ser/De

`TargetData` is converted to a dictionary with `dataclasses.asdict`, and is serialized with Python's
built-in JSON library.
When testing, the data is read into a `dict`, converted to a `TargetData`,
and stored in the top level `INPUT_DATA` variable.
The current deserialization logic should be resilient to changes in the schema,
but requires that all fields contain ONLY types that can be
directly serialized/deserialized by `json.dumps`.
The acceptable types are those that make up
[`common.JsonType`](https://github.com/rust-lang/rust/blob/bf9944f0b8006b152ef4d5f408ae75a0dde3d044/src/etc/lldb_batchmode/common.py#L17)

Since the serialization/deserialization is decoupled from the debugger logic, we can easily switch
to an alternative format if we find a better alternative to json.

The conversion logic from the debugger's internal representation to our schema classes lives in
`from_$DEBUGGER.py`.

Once imported, `common` automatically deserializes any existing input data and [stores it in the
global variable `INPUT_DATA`](https://github.com/rust-lang/rust/blob/bf9944f0b8006b152ef4d5f408ae75a0dde3d044/src/etc/lldb_batchmode/common.py#L523).
This data is what we test against.

> [!NOTE]
> Special care was taken to prevent `lldb_batchmode` from importing `common` unless a `repr` command
> was actually processed. This saves us from reading/writing input data for tests that don't need
> it.

#### Format minutae

Since type information is unique and unchanging once the debug session has begun, types are only
stored once at the top level, and are referred to by name everywhere else.

Pointer values change from run to run.
To prevent mismatches, pointer variables do not store their value.
This is equivalent to the wildcard `[...]` used in `-check` directives.

`BlessMetadata` is included in `TargetData`, but is not tested against.
It exists solely as a record of how the test data was generated,
to help in diagnosing issues that may occur due to Python or the debugger changing versions.

### Entry point and `--bless`

Upon encountering a `repr` pseudo-command, `lldb_batchmode.main` dispatches to
`check_$DEBUGGER.check()`.
If the `--bless` option was specified, the variable is converted from
the in-memory representation to our equivalent schema class.
This includes the variable's type,
visualizers, children, the children's types, etc.
Once inserted into `TargetData`,
the variable is tested against the data that was just saved to `TargetData`.

If no exception or errors occurred and the `--bless` option was specified, `INPUT_DATA` is written
to the appropriate file just before `lldb_batchmode` exits.
If errors occur, `INPUT_DATA` is simply discarded.

Currently, the `repr` pseudo-command is checked for directly.
GDB and LLDB both support creating custom CLI commands via Python code.
In the future, `repr` may be implemented as a CLI command for one or both debuggers.

### Check logic

`check_$DEBUGGER.check` converts the debugger's variable object into a `Variable` object and
compares the two.
If any mismatches are found, further processing is done to report errors in a more helpful manner.
This means that errors are encountered and reported immediately, which has a number of advantages.
Most importantly, since the debugger state has not changed since the failure, and we
still have access to the debugger's variable object, we can poll the debugger for more information
to provide more useful error messages.

For example, LLDB can be a bit coy when it comes to reporting errors that occur within
synthetic/summary provider calls.
This is especially true when running the command within another
command, as the tests do by calling `script import lldb_batchmode; lldb_batchmode.main()` and
executing commands in that context.

When we encounter an error, we can import the appropriate summary provider, pass the variable object
to it, and print the exception ourselves.
We can also inspect the synthetic provider class to make
sure it implements all the mandatory functions.

Errors *do not* immediately end the test.
This is especially important now that a `--bless` option has been added.
`--bless` updates all of the input data, so we need to print all of the errors so the reader can
make an informed decision about whether or not there are further changes that need to be made.
We absolutely do not want people accidentally blessing bad data
purely because the first error happened to be an expected change.

Errors are printed directly to `stdout` to appear as visible output from the `repr` pseudo command.
There are [several error helper functions](https://github.com/rust-lang/rust/blob/e7b595554e664e6bd281c8cf881093d6c71bc0e1/src/etc/lldb_batchmode/common.py#L35-L51)
to keep formatting consistent.

> [!NOTE]
> When LLDB is running a `script` command, it does not print the Python interpreter's `stderr`.
> If the interpreter exits with an exception, it will print that, but none of the rest of `stderr`.
> Instead, if we decide we want to print to `stderr`, we can use the debugger's by calling
> `lldb.debugger.GetErrorFileHandle` which returns a Python `io.TextIoWrapper`.

If no errors occurred for a given variable, `$VAR_NAME ok` is printed to `stdout` for `compiletest`
to match against.

Before `lldb_batchmode` exits, one last check is done to ensure that all the types and variables
that were present in `INPUT_DATA` have been checked against.
If this check fails, the script reports the untested types/variables and exits with an error code.

# LLDB versioning

Apple distributes a fork of LLDB with Xcode that contains Swift support.
This fork of LLDB does not use the same versioning scheme as LLVM's LLDB:

```
# Apple:
lldb-1703.0.236.21 Apple Swift version 6.2.3 (swiftlang-6.2.3.3.21 clang-1700.6.3.2)
# LLVM:
lldb version 22.1.2 (https://github.com/llvm/llvm-project revision 1ab49a973e210e97d61e5db6557180dcb92c3e98)
  clang revision 1ab49a973e210e97d61e5db6557180dcb92c3e98
  llvm revision 1ab49a973e210e97d61e5db6557180dcb92c3e98
```

It does not appear that the Apple LLDB's version is derived from LLVM's version, so we cannot easily
or automatically convert between the two.
Luckily, we can still check the base LLVM version manually
by checking the appropriate release branch in the Swift LLVM repo.
For our example above, no branch exists for `Swift 6.2.3`, but there is one for `6.2.2`.
The LLVM version is located in
[`llvm/utils/gn/secondary/llvm/version.gni`](https://github.com/swiftlang/llvm-project/blob/swift/release/6.2.2/llvm/utils/gn/secondary/llvm/version.gni).
As we can see from that example, the Apple LLDB version above corresponds to (roughly) LLVM LLDB
`19.1.5`.

This can be useful when diagnosing or writing new tests, as it allows us to get a better idea of
what features are available in the Apple LLDB used in CI.
For example, LLDB 19 was the first version to support Type Recognizer functions,
so we can assume our example Apple LLDB supports them.
