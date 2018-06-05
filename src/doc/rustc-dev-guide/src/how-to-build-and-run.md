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
# if you edit `trans`.
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

# Using the system allocator (instead of jemalloc) means that tools
# like valgrind and memcache work better.
use-jemalloc = false
```

### Running x.py and building a stage1 compiler

One thing to keep in mind is that `rustc` is a _bootstrapping_ compiler. That
is, since `rustc` is written in Rust, we need to use an older version of the
compiler to compile the newer version. In particular, the newer version of the
compiler, `libstd`, and other tooling may use some unstable features
internally. The result is the compiling `rustc` is done in stages.

- **Stage 0:** the stage0 compiler can be your existing
  (perhaps older version of)
  Rust compiler, the current _beta_ compiler or you may download the binary
  from the internet.
- **Stage 1:** the code in your clone (for new version)
  is then compiled with the stage0
  compiler to produce the stage1 compiler.
  However, it was built with an older compiler (stage0),
  so to optimize the stage1 compiler we go to next stage.
- **Stage 2:** we rebuild our stage1 compiler with itself
  to produce the stage2 compiler (i.e. it builds
  itself) to have all the _latest optimizations_.
- _(Optional)_ **Stage 3**: to sanity check of our new compiler,
  we can build it again
  with stage2 compiler which must be identical to itself,
  unless something has broken.

For hacking, often building the stage 1 compiler is enough, but for
final testing and release, the stage 2 compiler is used.

`./x.py check` is really fast to build the rust compiler.
It is, in particular, very useful when you're doing some kind of
"type-based refactoring", like renaming a method, or changing the
signature of some function.

Once you've created a config.toml, you are now ready to run
`x.py`. There are a lot of options here, but let's start with what is
probably the best "go to" command for building a local rust:

```bash
> ./x.py build -i --stage 1 src/libstd
```

What this command will do is the following:

- Using the beta compiler (also called stage 0), it will build the
  standard library and rustc from the `src` directory. The resulting
  compiler is called the "stage 1" compiler.
  - During this build, the `-i` (or `--incremental`) switch enables incremental
    compilation, so that if you later rebuild after editing things in
    `src`, you can save a bit of time.
- Using this stage 1 compiler, it will build the standard library.
  (this is what the `src/libstd`) means.

This is just a subset of the full rustc build. The **full** rustc build
(what you get if you just say `./x.py build`) has quite a few more steps:

- Build stage1 rustc with stage0 compiler.
- Build libstd with stage1 compiler (up to here is the same).
- Build rustc from `src` again, this time with the stage1 compiler
  (this part is new).
  - The resulting compiler here is called the "stage2" compiler.
- Build libstd with stage2 compiler.
- Build librustdoc and a bunch of other things.

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

### ctags

One of the challenges with rustc is that the RLS can't handle it, making code
navigation difficult. One solution is to use `ctags`. The following script can
be used to set it up: [https://github.com/nikomatsakis/rust-etags][etags].

CTAGS integrates into emacs and vim quite easily. The following can then be
used to build and generate tags:

```console
$ rust-ctags src/lib* && ./x.py build <something>
```

This allows you to do "jump-to-def" with whatever functions were around when
you last built, which is ridiculously useful.

[etags]: https://github.com/nikomatsakis/rust-etags
