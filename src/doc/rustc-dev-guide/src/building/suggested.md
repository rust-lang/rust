# Suggested Workflows

The full bootstrapping process takes quite a while. Here are three suggestions
to make your life easier.

## Check, check, and check again

The first workflow, which is useful
when doing simple refactorings, is to run `./x.py check`
continuously. Here you are just checking that the compiler can
**build**, but often that is all you need (e.g., when renaming a
method). You can then run `./x.py build` when you actually need to
run tests.

In fact, it is sometimes useful to put off tests even when you are not
100% sure the code will work. You can then keep building up
refactoring commits and only run the tests at some later time. You can
then use `git bisect` to track down **precisely** which commit caused
the problem. A nice side-effect of this style is that you are left
with a fairly fine-grained set of commits at the end, all of which
build and pass tests. This often helps reviewing.

## Incremental builds with `--keep-stage`.

Sometimes just checking
whether the compiler builds is not enough. A common example is that
you need to add a `debug!` statement to inspect the value of some
state or better understand the problem. In that case, you really need
a full build.  By leveraging incremental, though, you can often get
these builds to complete very fast (e.g., around 30 seconds). The only
catch is this requires a bit of fudging and may produce compilers that
don't work (but that is easily detected and fixed).

The sequence of commands you want is as follows:

- Initial build: `./x.py build -i --stage 1 src/libstd`
  - As [documented above](#command), this will build a functional
    stage1 compiler as part of running all stage0 commands (which include
    building a `libstd` compatible with the stage1 compiler) as well as the
    first few steps of the "stage 1 actions" up to "stage1 (sysroot stage1)
    builds libstd".
- Subsequent builds: `./x.py build -i --stage 1 src/libstd --keep-stage 1`
  - Note that we added the `--keep-stage 1` flag here

As mentioned, the effect of `--keep-stage 1` is that we just *assume* that the
old standard library can be re-used. If you are editing the compiler, this
is almost always true: you haven't changed the standard library, after
all.  But sometimes, it's not true: for example, if you are editing
the "metadata" part of the compiler, which controls how the compiler
encodes types and other states into the `rlib` files, or if you are
editing things that wind up in the metadata (such as the definition of
the MIR).

**The TL;DR is that you might get weird behavior from a compile when
using `--keep-stage 1`** -- for example, strange
[ICEs](../appendix/glossary.html) or other panics. In that case, you
should simply remove the `--keep-stage 1` from the command and
rebuild.  That ought to fix the problem.

You can also use `--keep-stage 1` when running tests. Something like this:

- Initial test run: `./x.py test -i --stage 1 src/test/ui`
- Subsequent test run: `./x.py test -i --stage 1 src/test/ui --keep-stage 1`

## Building with system LLVM

By default, LLVM is built from source, and that can take significant amount of
time.  An alternative is to use LLVM already installed on your computer.

This is specified in the `target` section of `config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
llvm-config = "/path/to/llvm/llvm-7.0.1/bin/llvm-config"
```

We have observed the following paths before, which may be different from your system:

- `/usr/bin/llvm-config-8`
- `/usr/lib/llvm-8/bin/llvm-config`

Note that you need to have the LLVM `FileCheck` tool installed, which is used
for codegen tests. This tool is normally built with LLVM, but if you use your
own preinstalled LLVM, you will need to provide `FileCheck` in some other way.
On Debian-based systems, you can install the `llvm-N-tools` package (where `N`
is the LLVM version number, e.g. `llvm-8-tools`).  Alternately, you can specify
the path to `FileCheck` with the `llvm-filecheck` config item in `config.toml`
or you can disable codegen test with the `codegen-tests` item in `config.toml`.
