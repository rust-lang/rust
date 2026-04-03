# aarch64-unknown-linux-pauthtest

**Tier: 3**

The target enables Pointer Authentication Code (PAC) support in Rust on AArch64
ELF based Linux systems using a pauthtest ABI (provided by LLVM) and
pauthtest-enabled sysroot with custom musl, serving as a reference libc
implementation.

Supported features include:
* authenticating signed function pointers for extern "C" function calls
  (corresponds to `-fptrauth-calls` included in pauthtest ABI as defined in
  LLVM)
* signing return address before spilling to stack and authenticating return
  address after restoring from stack for non-leaf functions (corresponds to
  `-fptrauth-returns`)
* Trapping if authentication failure is detected and FPAC feature is not present
  (corresponds to `-fptrauth-auth-traps`)
* Signing of init/fini array entries with the signing schema defined used for
  pauthtest ABI (corresponding to `-fptrauth-init-fini`,
  `-fptrauth-init-fini-address-discrimination`)
* Non-ABI-affecting indirect control flow hardening features included in
  pauthtest ABI (corresponding to `-faarch64-jump-table-hardening`,
  `-fptrauth-indirect-gotos`)
* signed ELF GOT entries (gated behind `-Z pauth_enable_elf_got`, off by
  default)

## Target maintainers

[@jchlanda](https://github.com/jchlanda)

## Requirements

This target supports cross-compilation from any Linux host, but execution
requires AArch64 with pointer authentication support (ARMv8.3 or higher).

## Standard library support

Full std support is available `core`, `alloc`, and `std` all build successfully.
All library tests (`core`, `alloc`, `std`) pass for this target as well.

## Building the target

Building the target itself requires pauthtest-enabled sysroot based on a custom
pauthtest-enabled musl to be present on the system. In order to fulfill the
requirement please follow the build instructions at [build scripts
repo](https://github.com/access-softek/pauth-toolchain-build-scripts). Please
provide the sysroot location via the `PAUTHTEST_SYSROOT` environment variable. A
Clang installation is also required. The Rust build system assumes that `clang`
is available on the `PATH`.

Introduction of `aarch64-unknown-linux-pauthtest` target needs to be propagated
to some crates, so that they can correctly recognise and handle it.
Specifically:
* `cc-rs`: https://github.com/jchlanda/cc-rs/tree/jakub/cc-v1.2.28-pauthtest
* `libc`: https://github.com/jchlanda/libc/tree/jakub/0.2.183-pauthtest
* `backtrace`: https://github.com/jchlanda/backtrace-rs/tree/jakub/backtrace-v0.3.76-pauthtest

The patched versions of `cc-rs` and `libc` will have to be registered through
`[patch.crates-io]` section of `Cargo.toml` files both in:
`<rust_root>/src/bootstrap/` and `<rust_root>/library/`. Check out `cc-rs` and
`libc` to `<rust_root>/patches` and update config files. See attached diff for
details:

```diff
diff --git a/library/Cargo.toml b/library/Cargo.toml
index e30e6240942..fb5a12f0065 100644
--- a/library/Cargo.toml
+++ b/library/Cargo.toml
@@ -59,3 +59,4 @@ rustflags = ["-Cpanic=abort"]
 rustc-std-workspace-core = { path = 'rustc-std-workspace-core' }
 rustc-std-workspace-alloc = { path = 'rustc-std-workspace-alloc' }
 rustc-std-workspace-std = { path = 'rustc-std-workspace-std' }
+libc = { path = '<rust_root>/patches/libc' }
diff --git a/src/bootstrap/Cargo.toml b/src/bootstrap/Cargo.toml
index e1725db60cf..46763cdf9a4 100644
--- a/src/bootstrap/Cargo.toml
+++ b/src/bootstrap/Cargo.toml
@@ -94,3 +94,6 @@ debug = 0
 [profile.dev.package]
 # Only use debuginfo=1 to further reduce compile times.
 bootstrap.debug = 1
+
+[patch.crates-io]
+cc = { path = '<rust_root>/patches/cc-rs' }
```

In contrast to `cc-rs` and `libc`, which are external crates resolved from
crates.io and can be overridden using [patch.crates-io], `backtrace` is included
in the Rust repository as a git submodule under:
`<rust_root>/library/backtrace`. At the time of writing the necessary change has
not yet been committed to it. Which means that for the time being it requires an
in-tree patch to be applied. The patch:

```diff
diff --git a/src/backtrace/libunwind.rs b/src/backtrace/libunwind.rs
index 0564f2e..a8a0d1a 100644
--- a/src/backtrace/libunwind.rs
+++ b/src/backtrace/libunwind.rs
@@ -79,6 +79,18 @@ impl Frame {
         // clause, and if this is fixed that test in theory can be run on macOS!
         if cfg!(target_vendor = "apple") {
             self.ip()
+        } else if cfg!(target_env = "pauthtest") {
+            // NOTE: As ip here is not signed (raw, non-PAC-enabled pointer) we
+            // must not use uw::_Unwind_FindEnclosingFunction. This is because,
+            // for pauthtest toolchain, libunwind will try to authenticate and
+            // resign it. Signing here (apart from risking creating a signing
+            // oracle) is not possible. According to the schema the value must
+            // be signed using SP as the discriminator - which is the problem.
+            // SP obtained here would not match the SP at the auth-resign time,
+            // as uw::_Unwind_FindEnclosingFunction creates a new context so
+            // the SP used for signing here would belong to a different frame
+            // that the one used for auth-resign. Hence return a raw value.
+            self.ip()
         } else {
             unsafe { uw::_Unwind_FindEnclosingFunction(self.ip()) }
         }
```

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["aarch64-unknown-linux-pauthtest"]
```

Make sure clang toolchain is included in `$PATH`, then add it to the
`bootstrap.toml`:

```toml
[target.aarch64-unknown-linux-pauthtest]
linker = "clang"
```

## Building Rust programs

Rust does not currently ship precompiled artifacts for this target. Programs
must be built using a locally compiled Rust toolchain. All programs must be
dynamically linked against musl from the PAC toolchain, using provided
interpreter:

```sh
<toolchain_install_dir>/aarch64-linux-pauthtest/usr/lib/libc.so
```

For a comprehensive example of how to interact between C and Rust programs
please consult `<rust_root>/tests/run-make/pauth-quicksort-c-driver/rmake.rs`,
the test builds a C executable linked against Rust library.
`<rust_root>/tests/run-make/pauth-quicksort-rust-driver/rmake.rs` shows how to
link a Rust program against a library compiled from a C source file.

## Cross-compilation

This target can be cross-compiled from any Linux based host, but execution must
take place on PAC aware AArch64 system.

## Testing

This target can be tested as normal with `x.py`.
The following categories are supported (all present in tree):
* Assembly tests
  * targets-aarch64_unknown_linux_pauthtest.rs
* LLVM IR/codegen tests
  * pauth-extern-c.rs
  * pauth-extern-c-direct-indirect-call.rs
  * pauth-init-fini.rs
  * pauth-attr-special-funcs.rs
* End-to-end execution tests
  * Rust-driven quicksort (pauth-quicksort-rust-driver)
  * C-driven quicksort (pauth-quicksort-c-driver)

All tests from `ui` and `library` subsets are expected to pass.

Command to run all passing tests:

```sh
x.py test --target aarch64-unknown-linux-pauthtest --force-rerun \
  library ui \
  tests/run-make/pauth-quicksort-rust-driver \
  tests/run-make/pauth-quicksort-c-driver \
  tests/codegen-llvm/pauth-attr-special-funcs.rs \
  tests/codegen-llvm/pauth-extern-c-direct-indirect-call.rs \
  tests/codegen-llvm/pauth-init-fini.rs \
  tests/codegen-llvm/pauth-extern-c.rs \
  tests/assembly-llvm/targets/targets-aarch64_unknown_linux_pauthtest.rs
```

## Cross-compilation toolchains and C code

This target supports interoperability with C code. Use the PAC-enabled LLVM
sysroot, described in Building the target section of this document. C code must
be compiled with the pauthtest aware compiler (a recent version of clang is
sufficient). Mixed Rust/C programs are supported and tested (e.g. quicksort
examples). Pointer authentication semantics must be consistent across Rust and C
components. The target only supports dynamic linking with the custom
interpreter.

## Limitation
Operand bundles should only be attached to indirect function calls. However, as
function pointer signing is unstable, we end up signing too eagerly (including
functions used for direct calls), hence operand bundles are added to all calls.
The issue is further discussed in a ticket at [rust-lang issue
tracker](https://github.com/rust-lang/rust/issues/152532).
