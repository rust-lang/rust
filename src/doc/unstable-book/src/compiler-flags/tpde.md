# `tpde`

The tracking issue for this feature is: None.

---

The `-Ztpde[=VALUE]` compiler flag controls whether TPDE-LLVM is used as the codegen backend for debug builds instead of LLVM. If not set, LLVM is used.

Possible values are:

- `try` (default): attempt to compile with TPDE-LLVM, but fall back to LLVM if unsupported IR is encountered
- `only`: attempt to compile with TPDE-LLVM, and bail if unsupported IR is encountered
