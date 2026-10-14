# Debugging TPDE

Bug reports and any other contributions are always welcome.
Do note that this backend is experimental, so there are many features missing at the moment.
Please ask on [Zulip](https://rust-lang.zulipchat.com/) for help if you get stuck or if our documentation is unclear.

# Reporting crashes

When using `-Z tpde=only`, you will likely see modules failing to compile due to certain LLVM IR features being unsupported by TPDE. An LLVM module should never fail to compile with `-Z tpde=try`, so if it does, this is a bug we would particularly appreciate knowing about.

It is also conceivable that a program compiled with TPDE will compile successfully but crash at runtime due to an ABI mismatch with LLVM or other miscompilation. These bugs can be similarly minimized and reported, and are of high interest to the maintainers of the `rustc` TPDE integration. These cases are harder to debug, so your help is highly appreciated.

# LLVM IR reproducer

A failure to compile can usually be traced back to an unsupported IR feature. You can enable logging by setting the `TPDE_LOGGING` option to `ON` in `src/bootstrap/src/core/build_steps/llvm.rs`, which will print a message containing the offending IR.

Having a minimal reproducer using this feature is extremely helpful. As TPDE-LLVM operates on LLVM IR, you can follow the instructions at [Debugging LLVM](../backend/debugging.md) to emit LLVM IR and minimize the Rust reproducer.

# Report your bug

- If the issue is due to an unsupported feature in TPDE, please create an issue at [https://github.com/tpde2/tpde/issues](https://github.com/tpde2/tpde/issues).
  - The goal of TPDE is to support the most commonly used subset of LLVM IR features, so it is possible that the feature will not be added in the interest of avoiding excessive complexity.
- If the issue is due to Rust emitting invalid LLVM IR or using TPDE incorrectly, please create an issue at [https://github.com/rust-lang/rust/issues](https://github.com/rust-lang/rust/issues).
- If unsure, please ask on Zulip before opening an issue.
- Please feel free to also add your rust code or a link to it.
