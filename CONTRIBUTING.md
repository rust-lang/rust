# Contributing to rust_codegen_gcc

Welcome to the rust_codegen_gcc project! This guide will help you get started as a contributor. The project aims to provide a GCC codegen backend for rustc, allowing Rust compilation on platforms unsupported by LLVM and potentially improving runtime performance through GCC's optimizations.

## Getting Started

### Setting Up Your Development Environment

1. Install the required dependencies:
   - rustup (follow instructions on the [official website](https://rustup.rs))
   - DejaGnu (for running libgccjit test suite)
   - Additional packages: `flex`, `libmpfr-dev`, `libgmp-dev`, `libmpc3`, `libmpc-dev`

2. Clone and configure the repository:
   ```bash
   git clone https://github.com/rust-lang/rust_codegen_gcc
   cd rust_codegen_gcc
   cp config.example.toml config.toml
   ```

3. Build the project:
   ```bash
   ./y.sh prepare  # downloads and patches sysroot
   ./y.sh build --sysroot --release
   ```

### Running Tests

To verify your setup:
```bash
# Run the full test suite
./y.sh test --release

# Test with a simple program
./y.sh cargo build --manifest-path tests/hello-world/Cargo.toml
```

## Communication Channels

- Matrix: Join our [Matrix channel](https://matrix.to/#/#rustc_codegen_gcc:matrix.org)
- IRC: Join us on [IRC](https://web.libera.chat/#rustc_codegen_gcc)
- GitHub Issues: For bug reports and feature discussions

We encourage new contributors to join our communication channels and introduce themselves. Feel free to ask questions about where to start or discuss potential contributions.

## Understanding Core Concepts

### Project Structure

The project consists of several key components:
- The GCC backend integration through libgccjit
- Rust compiler interface
- Test infrastructure

### Common Development Tasks

#### Running Specific Tests
To run a specific test:
1. Individual test: `./y.sh test --test <test_name>`
2. libgccjit tests: 
   ```bash
   cd gcc-build/gcc
   make check-jit
   # For a specific test:
   make check-jit RUNTESTFLAGS="-v -v -v jit.exp=jit.dg/test-asm.cc"
   ```

#### Debugging Tools
The project provides several environment variables for debugging:
- `CG_GCCJIT_DUMP_MODULE`: Dumps a specific module
- `CG_GCCJIT_DUMP_TO_FILE`: Creates C-like representation
- `CG_GCCJIT_DUMP_RTL`: Shows Register Transfer Language output

Full list of debugging options can be found in the README.

## Making Contributions

### Finding Issues to Work On
1. Look for issues labeled with `good-first-issue` or `help-wanted`
2. Check the project roadmap for larger initiatives
3. Consider improving documentation or tests

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
  - Common errors
  - Debugging GCC LTO
  - Git subtree sync
  - Sending patches to GCC

## Getting Help

If you're stuck or unsure about anything:
1. Check the existing documentation in the `doc/` directory
2. Ask in the IRC or Matrix channels
3. Open a GitHub issue for technical problems
4. Comment on the issue you're working on if you need guidance

Remember that all contributions, including documentation improvements, bug reports, and feature requests, are valuable to the project.