Generate and run programs using equivalent C and Rust intrinsics, checking that
each produces the same result from random inputs.

# Usage
```
USAGE:
    intrinsic-test [FLAGS] [OPTIONS] <INPUT>

FLAGS:
        --a32              Run tests for A32 intrinsics instead of A64
        --generate-only    Regenerate test programs, but don't build or run them
    -h, --help             Prints help information
    -V, --version          Prints version information

OPTIONS:
        --cppcompiler <CPPCOMPILER>    The C++ compiler to use for compiling the c++ code [default: clang++]
        --runner <RUNNER>              Run the C programs under emulation with this command
        --skip <SKIP>                  Filename for a list of intrinsics to skip (one per line)
        --toolchain <TOOLCHAIN>        The rust toolchain to use for building the rust code

ARGS:
    <INPUT>    The input file containing the intrinsics
```
