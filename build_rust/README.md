The Rust Build System
=====================

The Rust Build System is written in Rust and is managed as a Cargo package.
Building the latest Rust compiler is as simple as running:

```sh
$ cargo run
```

under this directory (where the build system is). The Rust Build System will
automatically build all supporting libraries (including LLVM) and bootstrap
a working stage2 compiler.

To speed up the build process by running parallel jobs, use `--nproc`:

```sh
$ cargo run -- --nproc=4
```

This will run 4 parallel jobs when building LLVM.

To show the command output during the build process, use `--verbose`:

```sh
$ cargo run -- --verbose
```

You can use `--no-bootstrap`, `--no-rebuild-llvm`, etc to control the build
process.

This build system supports out-of-tree build. Use `--rustc-root=<DIR>` to
specify the location of the source repo. Use `--build-dir=<DIR>` to specify
the root build directory.

Use `--help` to see a list of supported command line arguments.
