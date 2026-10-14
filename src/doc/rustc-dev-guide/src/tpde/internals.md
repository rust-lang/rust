TPDE-LLVM is an alternative LLVM codegen backend which aims to compile code significantly faster than LLVM while maintaining performance comparable to LLVM with `-O0`. It was introduced as part of a 2026 project goal, A[dd an experimental TPDE compilation option](https://goals.rust-lang.org/2026/tpde.html).

`rustc` has an experimental TPDE-LLVM backend option which can be used to improve compile times for debug builds. TPDE is currently missing the "long tail" of features supported by LLVM, but `rustc` supports falling back to regular LLVM codegen if TPDE fails.

# Supported `RUSTFLAGS`

To control the usage of TPDE-LLVM as the codegen backend, use the `-Z tpde` rustc flag (which can be passed to cargo via `RUSTFLAGS`). The valid options are:

- `<not set>`: use LLVM normally as the codegen backend. TPDE is not used
- `-Ztpde`, `-Ztpde=try`: attempt to compile with TPDE-LLVM, but fall back to LLVM if unsupported IR is encountered
- `-Ztpde=only`: attempt to compile with TPDE-LLVM, and bail if unsupported IR is encountered
