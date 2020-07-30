# Suggested Workflows

The full bootstrapping process takes quite a while. Here are five suggestions
to make your life easier.

## Configuring `rust-analyzer` for `rustc`

`rust-analyzer` can help you check and format your code whenever you save
a file. By default, `rust-analyzer` runs the `cargo check` and `rustfmt`
commands, but you can override these commands to use more adapted versions
of these tools when hacking on `rustc`. For example, for Visual Studio Code,
you can write:

```JSON
{
    "rust-analyzer.checkOnSave.overrideCommand": [
        "./x.py",
        "check",
        "--json-output"
    ],
    "rust-analyzer.rustfmt.overrideCommand": [
      "./build/TARGET_TRIPLE/stage0/bin/rustfmt"
    ],
    "editor.formatOnSave": true
}
```

in your `.vscode/settings.json` file. This will ask `rust-analyzer` to use
`x.py check` to check the sources, and the stage 0 rustfmt to format them.

## Check, check, and check again

When doing simple refactorings, it can be useful to run `./x.py check`
continuously. If you set up `rust-analyzer` as described above, this will
be done for you every time you save a file. Here you are just checking that
the compiler can **build**, but often that is all you need (e.g., when renaming a
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

- Initial build: `./x.py build -i library/std`
  - As [documented above](#command), this will build a functional
    stage1 compiler as part of running all stage0 commands (which include
    building a `std` compatible with the stage1 compiler) as well as the
    first few steps of the "stage 1 actions" up to "stage1 (sysroot stage1)
    builds std".
- Subsequent builds: `./x.py build -i library/std --keep-stage 1`
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
[ICEs](../appendix/glossary.html#ice) or other panics. In that case, you
should simply remove the `--keep-stage 1` from the command and
rebuild.  That ought to fix the problem.

You can also use `--keep-stage 1` when running tests. Something like this:

- Initial test run: `./x.py test -i src/test/ui`
- Subsequent test run: `./x.py test -i src/test/ui --keep-stage 1`

## Fine-tuning optimizations

Setting `optimize = false` makes the compiler too slow for tests. However, to
improve the test cycle, you can disable optimizations selectively only for the
crates you'll have to rebuild
([source](https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/incremental.20compilation.20question/near/202712165)).
For example, when working on `rustc_mir_build`, the `rustc_mir_build` and
`rustc_driver` crates take the most time to incrementally rebuild. You could
therefore set the following in the root `Cargo.toml`:
```toml
[profile.release.package.rustc_mir_build]
opt-level = 0
[profile.release.package.rustc_driver]
opt-level = 0
```

## Working on multiple branches at the same time

Working on multiple branches in parallel can be a little annoying, since
building the compiler on one branch will cause the old build and the
incremental compilation cache to be overwritten. One solution would be
to have multiple clones of the repository, but that would mean storing the
Git metadata multiple times, and having to update each clone individually.

Fortunately, Git has a better solution called [worktrees]. This lets you
create multiple "working trees", which all share the same Git database.
Moreover, because all of the worktrees share the same object database,
if you update a branch (e.g. master) in any of them, you can use the new
commits from any of the worktrees. One caveat, though, is that submodules
do not get shared. They will still be cloned multiple times.

[worktrees]: https://git-scm.com/docs/git-worktree

Given you are inside the root directory for your rust repository, you can
create a "linked working tree" in a new "rust2" directory by running
the following command:

```bash
git worktree add ../rust2
```

Creating a new worktree for a new branch based on `master` looks like:

```bash
git worktree add -b my-feature ../rust2 master
```

You can then use that rust2 folder as a separate workspace for modifying
and building `rustc`!

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
