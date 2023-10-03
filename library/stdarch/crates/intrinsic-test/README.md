Generate and run programs using equivalent C and Rust intrinsics, checking that
each produces the same result from random inputs.

# Usage
```
USAGE:
    intrinsic-test [FLAGS] [OPTIONS] <INPUT>

FLAGS:
        --a32        Run tests for A32 instrinsics instead of A64
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --cppcompiler <CPPCOMPILER>    The C++ compiler to use for compiling the c++ code [default: clang++]
        --runner <RUNNER>              Run the C programs under emulation with this command
        --skip <SKIP>                  Filename for a list of intrinsics to skip (one per line)
        --toolchain <TOOLCHAIN>        The rust toolchain to use for building the rust code

ARGS:
    <INPUT>    The input file containing the intrinsics
```

The intrinsic.csv is the arm neon tracking google sheet (https://docs.google.com/spreadsheets/d/1MqW1g8c7tlhdRWQixgdWvR4uJHNZzCYAf4V0oHjZkwA/edit#gid=0)
that contains the intrinsic list. The done percentage column should be renamed to "enabled".

