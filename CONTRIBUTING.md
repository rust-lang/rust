# Contributing to rust_codegen_gcc

Welcome to the `rust_codegen_gcc` project! This guide will help you get started as a contributor. The project aims to provide a GCC codegen backend for rustc, allowing Rust compilation on platforms unsupported by LLVM and potentially improving runtime performance through GCC's optimizations.

## Getting Started

### Setting Up Your Development Environment

For detailed setup instructions including dependencies, build steps, and initial testing, please refer to our [README](https://github.com/rust-lang/rustc_codegen_gcc/blob/master/Readme.md). The README contains the most up-to-date information on:

- Required dependencies and system packages
- Repository setup and configuration
- Build process
- Basic test verification

Once you've completed the setup process outlined in the README, you can proceed with the contributor-specific information below.

## Communication Channels

- Matrix: Join our [Matrix channel](https://matrix.to/#/#rustc_codegen_gcc:matrix.org)
- IRC: Join us on [IRC](https://web.libera.chat/#rustc_codegen_gcc)
- GitHub Issues: For bug reports and feature discussions

We encourage new contributors to join our communication channels and introduce themselves. Feel free to ask questions about where to start or discuss potential contributions.

## Understanding Core Concepts

### Common Development Tasks

#### Running Specific Tests
To run specific tests, use appropriate flags such as:
- `./y.sh test --test-libcore`
- `./y.sh test --std-tests`
- `cargo test -- <name of test>`

Additionally, you can run the tests of `libgccjit`:
```bash
# libgccjit tests
cd gcc-build/gcc
make check-jit
# For a specific test:
make check-jit RUNTESTFLAGS="-v -v -v jit.exp=jit.dg/test-asm.cc"
```

#### Debugging Tools
The project provides several environment variables for debugging:
- `CG_GCCJIT_DUMP_GIMPLE`: Most commonly used debug dump
- `CG_RUSTFLAGS`: Additional Rust compiler flags
- `CG_GCCJIT_DUMP_MODULE`: Dumps a specific module
- `CG_GCCJIT_DUMP_TO_FILE`: Creates C-like representation

Full list of debugging options can be found in the [README](/rust-lang/rustc_codegen_gcc#env-vars).

## Making Contributions

### Finding Issues to Work On
1. Look for issues labeled with [`good first issue`](/rust-lang/rustc_codegen_gcc/issues?q=is%3Aissue state%3Aopen label%3A"good first issue") or [`help wanted`](/rust-lang/rustc_codegen_gcc/issues?q=is%3Aissue state%3Aopen label%3A"help wanted")
2. Check the [progress report](https://blog.antoyo.xyz/rustc_codegen_gcc-progress-report-34#state_of_rustc_codegen_gcc) for larger initiatives
3. Consider improving documentation or investigating [failing tests](https://github.com/rust-lang/rustc_codegen_gcc/tree/master/tests)(except `failing-ui-tests12.txt`)

### Pull Request Process
1. Fork the repository and create a new branch
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a PR with a description of your changes

### Code Style Guidelines
- Follow Rust standard coding conventions
- Ensure your code passes `rustfmt` and `clippy`
- Add comments explaining complex logic, especially in GCC interface code

## Additional Resources

- [Rustc Dev Guide](https://rustc-dev-guide.rust-lang.org/)
- [GCC Internals Documentation](https://gcc.gnu.org/onlinedocs/gccint/)
- Project-specific documentation in the `doc/` directory:
  - [Common errors](/rust-lang/rustc_codegen_gcc/blob/master/doc/errors.md)
  - [Debugging](/rust-lang/rustc_codegen_gcc/blob/master/doc/debugging.md)
  - [Debugging libgccjit](/rust-lang/rustc_codegen_gcc/blob/master/doc/debugging-libgccjit.md)
  - [Git subtree sync](/rust-lang/rustc_codegen_gcc/blob/master/doc/subtree.md)
  - [List of useful commands](/rust-lang/rustc_codegen_gcc/blob/master/doc/tips.md)
  - [Send a patch to GCC](/rust-lang/rustc_codegen_gcc/blob/master/doc/sending-gcc-patch.md)

## Getting Help

If you're stuck or unsure about anything:
1. Check the existing documentation in the `doc/` directory
2. Ask in the IRC or Matrix channels
3. Open a GitHub issue for technical problems
4. Comment on the issue you're working on if you need guidance

Remember that all contributions, including documentation improvements, bug reports, and feature requests, are valuable to the project.
