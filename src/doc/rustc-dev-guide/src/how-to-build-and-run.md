# How to build the compiler and run what you built

The compiler is built using a tool called `x.py`. You will need to
have Python installed to run it. But before we get to that, if you're going to
be hacking on `rustc`, you'll want to tweak the configuration of the compiler.
The default configuration is oriented towards running the compiler as a user,
not a developer.

### Create a config.toml

To start, copy [`config.toml.example`] to `config.toml`:

[`config.toml.example`]: https://github.com/rust-lang/rust/blob/master/config.toml.example

```bash
> cd $RUST_CHECKOUT
> cp config.toml.example config.toml
```

Then you will want to open up the file and change the following
settings (and possibly others, such as `llvm.ccache`):

```toml
[llvm]
# Enables LLVM assertions, which will check that the LLVM bitcode generated
# by the compiler is internally consistent. These are particularly helpful
# if you edit `codegen`.
assertions = true

[rust]
# This enables some assertions, but more importantly it enables the `debug!`
# logging macros that are essential for debugging rustc.
debug-assertions = true

# This will make your build more parallel; it costs a bit of runtime
# performance perhaps (less inlining) but it's worth it.
codegen-units = 0

# I always enable full debuginfo, though debuginfo-lines is more important.
debuginfo = true

# Gives you line numbers for backtraces.
debuginfo-lines = true
```

### What is x.py?

x.py is the script used to orchestrate the tooling in the rustc repository.
It is the script that can build docs, run tests, and compile rustc.
It is the now preferred way to build rustc and it replaces the old makefiles
from before. Below are the different ways to utilize x.py in order to
effectively deal with the repo for various common tasks.

### Running x.py and building a stage1 compiler

One thing to keep in mind is that `rustc` is a _bootstrapping_
compiler. That is, since `rustc` is written in Rust, we need to use an
older version of the compiler to compile the newer version. In
particular, the newer version of the compiler, `libstd`, and other
tooling may use some unstable features internally. The result is that
compiling `rustc` is done in stages:

- **Stage 0:** the stage0 compiler is usually the current _beta_ compiler
  (`x.py` will download it for you); you can configure `x.py` to use something
  else, though.
- **Stage 1:** the code in your clone (for new version) is then
  compiled with the stage0 compiler to produce the stage1 compiler.
  However, it was built with an older compiler (stage0), so to
  optimize the stage1 compiler we go to next stage.
  - (In theory, the stage1 compiler is functionally identical to the
    stage2 compiler, but in practice there are subtle differences. In
    particular, the stage1 compiler itself was built by stage0 and
    hence not by the source in your working directory: this means that
    the symbol names used in the compiler source may not match the
    symbol names that would have been made by the stage1 compiler.
    This can be important when using dynamic linking (e.g., with
    derives. Sometimes this means that some tests don't work when run
    with stage1.)
- **Stage 2:** we rebuild our stage1 compiler with itself to produce
  the stage2 compiler (i.e. it builds itself) to have all the _latest
  optimizations_. (By default, we copy the stage1 libraries for use by
  the stage2 compiler, since they ought to be identical.)
- _(Optional)_ **Stage 3**: to sanity check of our new compiler, we
  can build the libraries with the stage2 compiler. The result ought
  to be identical to before, unless something has broken.

#### Build Flags

There are other flags you can pass to the build portion of x.py that can be
beneficial to cutting down compile times or fitting other things you might
need to change. They are:

```bash
Options:
    -v, --verbose       use verbose output (-vv for very verbose)
    -i, --incremental   use incremental compilation
        --config FILE   TOML configuration file for build
        --build BUILD   build target of the stage0 compiler
        --host HOST     host targets to build
        --target TARGET target targets to build
        --on-fail CMD   command to run on failure
        --stage N       stage to build
        --keep-stage N  stage to keep without recompiling
        --src DIR       path to the root of the rust checkout
    -j, --jobs JOBS     number of jobs to run in parallel
    -h, --help          print this help message
```

For hacking, often building the stage 1 compiler is enough, but for
final testing and release, the stage 2 compiler is used.

`./x.py check` is really fast to build the rust compiler.
It is, in particular, very useful when you're doing some kind of
"type-based refactoring", like renaming a method, or changing the
signature of some function.

<a name=command></a>

Once you've created a config.toml, you are now ready to run
`x.py`. There are a lot of options here, but let's start with what is
probably the best "go to" command for building a local rust:

```bash
> ./x.py build -i --stage 1 src/libstd
```

This may *look* like it only builds libstd, but that is not the case.
What this command does is the following:

- Build libstd using the stage0 compiler (using incremental)
- Build librustc using the stage0 compiler (using incremental)
  - This produces the stage1 compiler
- Build libstd using the stage1 compiler (cannot use incremental)

This final product (stage1 compiler + libs built using that compiler)
is what you need to build other rust programs.

Note that the command includes the `-i` switch. This enables incremental
compilation. This will be used to speed up the first two steps of the process:
in particular, if you make a small change, we ought to be able to use your old
results to make producing the stage1 **compiler** faster.

Unfortunately, incremental cannot be used to speed up making the
stage1 libraries.  This is because incremental only works when you run
the *same compiler* twice in a row.  In this case, we are building a
*new stage1 compiler* every time. Therefore, the old incremental
results may not apply. **As a result, you will probably find that
building the stage1 libstd is a bottleneck for you** -- but fear not,
there is a (hacky) workaround.  See [the section on "recommended
workflows"](#workflow) below.

Note that this whole command just gives you a subset of the full rustc
build. The **full** rustc build (what you get if you just say `./x.py
build`) has quite a few more steps:

- Build librustc and rustc with the stage1 compiler.
  - The resulting compiler here is called the "stage2" compiler.
- Build libstd with stage2 compiler.
- Build librustdoc and a bunch of other things with the stage2 compiler.

<a name=toolchain></a>

### Build specific components

   Build only the libcore library

```bash
> ./x.py build src/libcore
```

   Build the libcore and libproc_macro library only

```bash
> ./x.py build src/libcore src/libproc_macro
```

   Build only libcore up to Stage 1

```bash
> ./x.py build src/libcore --stage 1
```

Sometimes you might just want to test if the part you’re working on can
compile. Using these commands you can test that it compiles before doing
a bigger build to make sure it works with the compiler. As shown before
you can also pass flags at the end such as --stage.


### Creating a rustup toolchain

Once you have successfully built rustc, you will have created a bunch
of files in your `build` directory. In order to actually run the
resulting rustc, we recommend creating rustup toolchains. The first
one will run the stage1 compiler (which we built above). The second
will execute the stage2 compiler (which we did not build, but which
you will likely need to build at some point; for example, if you want
to run the entire test suite).

```bash
> rustup toolchain link stage1 build/<host-triple>/stage1
> rustup toolchain link stage2 build/<host-triple>/stage2
```

The `<host-triple>` would typically be one of the following:

- Linux: `x86_64-unknown-linux-gnu`
- Mac: `x86_64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`

Now you can run the rustc you built with. If you run with `-vV`, you
should see a version number ending in `-dev`, indicating a build from
your local environment:

```bash
> rustc +stage1 -vV
rustc 1.25.0-dev
binary: rustc
commit-hash: unknown
commit-date: unknown
host: x86_64-unknown-linux-gnu
release: 1.25.0-dev
LLVM version: 4.0
```

<a name=workflow></a>

### Suggested workflows for faster builds of the compiler

There are two workflows that are useful for faster builds of the
compiler.

**Check, check, and check again.** The first workflow, which is useful
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

**Incremental builds with `--keep-stage`.** Sometimes just checking
whether the compiler builds is not enough. A common example is that
you need to add a `debug!` statement to inspect the value of some
state or better understand the problem. In that case, you really need
a full build.  By leveraging incremental, though, you can often get
these builds to complete very fast (e.g., around 30 seconds): the only
catch is this requires a bit of fudging and may produce compilers that
don't work (but that is easily detected and fixed).

The sequence of commands you want is as follows:

- Initial build: `./x.py build -i --stage 1 src/libstd`
  - As [documented above](#command), this will build a functional
    stage1 compiler
- Subsequent builds: `./x.py build -i --stage 1 src/libstd --keep-stage 1`
  - Note that we added the `--keep-stage 1` flag here

The effect of `--keep-stage 1` is that we just *assume* that the old
standard library can be re-used. If you are editing the compiler, this
is almost always true: you haven't changed the standard library, after
all.  But sometimes, it's not true: for example, if you are editing
the "metadata" part of the compiler, which controls how the compiler
encodes types and other states into the `rlib` files, or if you are
editing things that wind up in the metadata (such as the definition of
the MIR).

**The TL;DR is that you might get weird behavior from a compile when
using `--keep-stage 1`** -- for example, strange
[ICEs](appendix/glossary.html) or other panics. In that case, you
should simply remove the `--keep-stage 1` from the command and
rebuild.  That ought to fix the problem.

You can also use `--keep-stage 1` when running tests. Something like
this:

- Initial test run: `./x.py test -i --stage 1 src/test/ui`
- Subsequent test run: `./x.py test -i --stage 1 src/test/ui --keep-stage 1`

### Other x.py commands

Here are a few other useful x.py commands. We'll cover some of them in detail
in other sections:

- Building things:
  - `./x.py clean` – clean up the build directory (`rm -rf build` works too,
    but then you have to rebuild LLVM)
  - `./x.py build --stage 1` – builds everything using the stage 1 compiler,
    not just up to libstd
  - `./x.py build` – builds the stage2 compiler
- Running tests (see the [section on running tests](./tests/running.html) for
  more details):
  - `./x.py test --stage 1 src/libstd` – runs the `#[test]` tests from libstd
  - `./x.py test --stage 1 src/test/run-pass` – runs the `run-pass` test suite
  - `./x.py test --stage 1 src/test/ui/const-generics` - runs all the tests in
  the `const-generics/` subdirectory of the `ui` test suite
  - `./x.py test --stage 1 src/test/ui/const-generics/const-types.rs` - runs
  the single test `const-types.rs` from the `ui` test suite

### ctags

One of the challenges with rustc is that the RLS can't handle it, since it's a
bootstrapping compiler. This makes code navigation difficult. One solution is to
use `ctags`.

`ctags` has a long history and several variants. Exhuberant CTags seems to be
quite commonly distributed but it does not have out-of-box Rust support. Some
distributions seem to use [Universal Ctags][utags], which is a maintained fork
and does have built-in Rust support.

The following script can be used to set up Exhuberant Ctags:
[https://github.com/nikomatsakis/rust-etags][etags].

`ctags` integrates into emacs and vim quite easily. The following can then be
used to build and generate tags:

```console
$ rust-ctags src/lib* && ./x.py build <something>
```

This allows you to do "jump-to-def" with whatever functions were around when
you last built, which is ridiculously useful.

[etags]: https://github.com/nikomatsakis/rust-etags
[utags]: https://github.com/universal-ctags/ctags

### Cleaning out build directories

Sometimes you need to start fresh, but this is normally not the case.
If you need to run this then rustbuild is most likely not acting right and
you should file a bug as to what is going wrong. If you do need to clean
everything up then you only need to run one command!

   ```bash
   > ./x.py clean
   ```

### Compiler Documentation

The documentation for the rust components are found at [rustc doc].

[rustc doc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/
