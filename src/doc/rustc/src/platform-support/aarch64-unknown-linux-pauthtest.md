# aarch64-unknown-linux-pauthtest

**Tier: 3**

The target enables Pointer Authentication Code (PAC) support in Rust on AArch64
ELF based Linux systems using a pauthtest ABI (provided by LLVM) and
pauthtest-enabled sysroot with custom musl, serving as a reference libc
implementation. It requires dynamic linking with a pauthtest-enabled dynamic
linker serving as ELF interpreter capable of resolving pauth relocations and
respecting pauthtest ABI constraints.

Supported features include:
* authenticating signed function pointers for extern "C" function calls
  (corresponds to `-fptrauth-calls` included in pauthtest ABI as defined in
  LLVM)
* signing return address before spilling to stack and authenticating return
  address after restoring from stack for non-leaf functions (corresponds to
  `-fptrauth-returns`)
* trapping if authentication failure is detected and FPAC feature is not present
  (corresponds to `-fptrauth-auth-traps`)
* signing of init/fini array entries with the signing schema used for pauthtest
  ABI (corresponding to `-fptrauth-init-fini`,
  `-fptrauth-init-fini-address-discrimination`)
* non-ABI-affecting indirect control flow hardening features included in
  pauthtest ABI (corresponding to `-faarch64-jump-table-hardening`,
  `-fptrauth-indirect-gotos`)
* signed ELF GOT entries (gated behind `-Z ptrauth-elf-got`, off by default)

A tracking issue for adding support for the AArch64 pointer authentication ABI
in Rust can be found at
[#148640](https://github.com/rust-lang/rust/issues/148640).

Existing compiler support, such as enabling branch authentication instructions
(i.e.: `-Z branch-protection`) provide limited functionality, mainly signing
return addresses (`pac-ret`). The new target goes further by enabling ABI-level
pointer authentication support.

## Target maintainers

[@jchlanda](https://github.com/jchlanda)

## Requirements

This target supports cross-compilation from any Linux host, but execution
requires AArch64 with pointer authentication support (ARMv8.3-A or higher).

## Standard library support

Full std support is available: `core`, `alloc`, and `std` all build
successfully. All library tests (`core`, `alloc`, `std`) pass for this target as
well.

## Building the toolchain

Building this target requires a pauthtest-enabled sysroot based on a custom musl
toolchain. The sysroot must be available on the system before compilation. To
build it, follow the instructions in the [build scripts
repo](https://github.com/access-softek/pauth-toolchain-build-scripts).

The target uses Clang, please make sure it is `v22.1.0` or higher. When using a
system-provided Clang, a compiler wrapper is required to supply the necessary
flags. Please consult the listing:

```sh
#!/usr/bin/env sh

clang \
  -target aarch64-unknown-linux-pauthtest \
  -march=armv8.3-a+pauth \
  --sysroot <toolchain_root>/aarch64-linux-pauthtest/usr \
  -resource-dir <toolchain_root>/lib/clang/<version> \
  --rtlib=compiler-rt \
  --ld-path=/usr/bin/ld.lld \
  --unwindlib=libunwind \
  -Wl,--dynamic-linker=<toolchain_root>/aarch64-linux-pauthtest/usr/lib/libc.so \
  -Wl,--rpath=<toolchain_root>/aarch64-linux-pauthtest/usr/lib \
  "$@"
```

The Rust compiler validates the name of the configured C compiler, so when using
a wrapper its name must contain `clang`. A recommended name is
`aarch64-unknown-linux-pauthtest-clang`. Update the script to set `--sysroot`,
`-resource-dir`, `--dynamic-linker` and `--rpath` correctly by replacing
`<toolchain_root>` with the directory produced by the build scripts and the
`<version>` with LLVM's version. Make the wrapper executable.

To verify that the toolchain layout is correct, check that:
* the sysroot contains a pauthtest-enabled version of libunwind
  (`<toolchain_root>/aarch64-linux-pauthtest/usr/lib/libunwind.so`),
* the Clang resource directory contains the appropriate `compiler-rt` objects
  (`<toolchain_root>/lib/clang/<version>/lib/aarch64-unknown-linux-pauthtest/{clang_rt.crtbegin.o,clang_rt.crtend.o}`)

When using the AccessSoftek scripts to build the sysroot, the result includes a
Clang-based toolchain. In this case, no wrapper script is required,
`<toolchain_root>/bin/aarch64-linux-pauthtest-clang` can be used directly.

## Building the target

Introduction of `aarch64-unknown-linux-pauthtest` target needs to be propagated
to various crates/repos, so that they can correctly recognise and handle it.
Specifically:
* `cc-rs`: https://github.com/jchlanda/cc-rs/tree/jakub/cc-v1.2.28-pauthtest
* `libc`: https://github.com/jchlanda/libc/tree/jakub/0.2.183-pauthtest
* `backtrace`: https://github.com/jchlanda/backtrace-rs/tree/jakub/backtrace-v0.3.76-pauthtest

The patched versions of `cc-rs` and `libc` will have to be registered through
`[patch.crates-io]` section of `Cargo.toml` files both in:
`<rust_root>/src/bootstrap/` and `<rust_root>/library/`. Check out `cc-rs` and
`libc` to `<rust_root>/patches` and update config files. See attached diff for
details:

<details>

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

</details>

In contrast to `cc-rs` and `libc`, which are external crates resolved from
[crates.io](https://crates.io/) and can be overridden using `[patch.crates-io]`,
`backtrace` is included in the Rust repository as a git submodule under
`<rust_root>/library/backtrace`. At the time of writing, the necessary change
has not yet been committed there, which means an in-tree patch is currently
required. The patch:

<details>

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

</details>

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["aarch64-unknown-linux-pauthtest"]
```

Specify the binaries used by the target.

```toml
[target.aarch64-unknown-linux-pauthtest]
cc = "<path_to>/aarch64-unknown-linux-pauthtest-clang"
ar = "<path_to>/llvm-ar"
ranlib = "<path_to>/llvm-ranlib"
linker = "<path_to>/aarch64-unknown-linux-pauthtest-clang"
```

Note that `cc` and `linker` must refer to the same binary (either Clang itself
or its wrapper). The bootstrap process will fail if they differ. On non-AArch64
systems, ensure that QEMU is installed and that `binfmt_misc` is correctly
configured so that foreign architecture binaries can be executed transparently.

## Building Rust programs

Rust does not currently ship precompiled artifacts for this target. Programs
must be built using a locally compiled Rust toolchain, with
`aarch64-unknown-linux-pauthtest` target enabled.

For a comprehensive example of how to interact between C and Rust programs
within the testing framework please consult
`<rust_root>/tests/run-make/pauth-quicksort-c-driver/rmake.rs`, the test builds
a C executable linked against Rust library.
`<rust_root>/tests/run-make/pauth-quicksort-rust-driver/rmake.rs` shows how to
link a Rust program against a library compiled from a C source file.

### Minimal standalone Rust and C interoperability example

A minimal standalone example demonstrating Rust and C interoperability on the
`aarch64-unknown-linux-pauthtest` target is listed below.

<details>

* Project structure

```text
rust_c_indirect/
  ┣━ Cargo.toml
  ┣━ build.rs
  ┣━ src/
  ┃  ┗━ main.rs
  ┣━ c_src/
  ┃  ┗━ plugin.c
  ┗━ target/
```

* `Cargo.toml`

```toml
[package]
name = "rust_c_indirect"
edition = "2024"
build = "build.rs"
```

* `build.rs`

```rust, ignore (platform-specific)
use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=c_src/plugin.c");

    let clang = "<path_to>/aarch64-unknown-linux-pauthtest-clang";

    let out_dir = env::var("OUT_DIR").unwrap();
    let lib_path = Path::new(&out_dir).join("libplugin.so");
    let c_src = "c_src/plugin.c";

    let status = Command::new(clang)
        .args(["-shared", "-fPIC", c_src])
        .arg("-o")
        .arg(&lib_path)
        .status()
        .unwrap_or_else(|_| panic!("failed to build shared library"));
    assert!(status.success(), "failed to build shared library");

    println!("cargo:rustc-link-arg=-Wl,--dynamic-linker=<toolchain_root>/aarch64-linux-pauthtest/usr/lib/libc.so");
    println!("cargo:rustc-link-arg=-Wl,-rpath,<toolchain_root>/aarch64-linux-pauthtest/usr/lib");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=dylib=plugin");
}

```

* `src/main.rs`

```rust, ignore (platform-specific)
use std::ptr;
use std::os::raw::c_int;

unsafe extern "C" {
    fn add(a: c_int, b: c_int) -> c_int;
}

static OP: unsafe extern "C" fn(c_int, c_int) -> c_int = add;

fn main() {
    let a = 10;
    let b = 32;

    let op = unsafe { ptr::read_volatile(&raw const OP) };
    let result = unsafe { op(a, b) };

    println!("Result: {}", result);
}
```

* `c_src/plugin.c`

```c
int add(int a, int b) { return a + b; }
```

</details>

* compile: `cargo build --target aarch64-unknown-linux-pauthtest --release`
* run: `./target/aarch64-unknown-linux-pauthtest/release/rust_c_indirect`

Please make sure that `LD_LIBRARY_PATH` points to the directory containing
`libplugin.so`. For example:
`LD_LIBRARY_PATH=./target/aarch64-unknown-linux-pauthtest/release/build/rust_c_indirect-<hash>/out/`.

To inspect pointer authentication behavior in IR, build with:
`RUSTFLAGS="--emit=llvm-ir"`. This generates an LLVM IR file, e.g.:
`target/aarch64-unknown-linux-pauthtest/release/deps/rust_c_indirect-*.ll`.
Relevant excerpt:

```llvm
@_RNvCscVIHJvJIt8C_15rust_c_indirect2OP = internal constant ptr ptrauth (ptr @add, i32 0), align 8

%0 = load volatile ptr, ptr @_RNvCscVIHJvJIt8C_15rust_c_indirect2OP, align 8, !nonnull !5, !noundef !5
%1 = tail call noundef i32 %0(i32 noundef 10, i32 noundef 32) #6 [ "ptrauth"(i32 0, i64 0) ]
```

Which shows that:
* function pointer (`@add`) is signed using `ptrauth`, when global variable is
  initialized,
* the call is performed indirectly via a signed pointer,
* the `ptrauth` operand bundle enforces authentication at call time.

Note, when building crates it is necessary to explicitly point Cargo to the
linker it has to use. This can be achieved by using a `config.toml` file (either
local to the project, or global), or by setting a
`CARGO_TARGET_AARCH64_UNKNOWN_LINUX_PAUTHTEST_LINKER` variable. For example:
* `.cargo/config.toml`

```toml
[target.aarch64-unknown-linux-pauthtest]
linker = "<path_to>/aarch64-unknown-linux-pauthtest-clang"
```

* `export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_PAUTHTEST_LINKER=<path_to>/aarch64-unknown-linux-pauthtest-clang`

Without it Cargo falls back to the system C toolchain (cc) and the compilation
fails.

## Cross-compilation toolchains and C code

This target supports interoperability with C code. Use the pauthtest-enabled
sysroot, described in building the toolchain section of this document. C code
must be compiled with the pauthtest aware compiler. Mixed Rust/C programs are
supported and tested (e.g. quicksort examples). Pointer authentication semantics
must be consistent across Rust and C components. The target only supports
dynamic linking.

The target can be cross-compiled from any Linux-based host, but execution
requires an AArch64 system that implements Pointer Authentication (PAC). In
practice, this means a CPU conforming to at least the Armv8.3-A architecture,
where the
[FEAT_PAuth](https://developer.arm.com/documentation/109697/2025_06/Feature-descriptions/The-Armv8-3-architecture-extension?lang=en#md448-the-armv83-architecture-extension__feat_FEAT_PAuth)
extension is defined.

Cross-compilation has been successfully performed on both
`aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu` hosts.

## Testing

This target can be tested as normal with `x.py`.
The following categories are supported (all present in tree):
* Assembly tests
  * targets-aarch64_unknown_linux_pauthtest.rs
* LLVM IR/codegen tests
  * pauth-extern-c.rs
  * pauth-extern-c-direct-indirect-call.rs
  * pauth-extern-weak-global.rs
  * pauth-init-fini.rs
  * pauth-attr-special-funcs.rs
* End-to-end execution tests
  * Rust-driven quicksort (pauth-quicksort-rust-driver)
  * C-driven quicksort (pauth-quicksort-c-driver)
* UI error reporting (pauthtest does not support `+crt-static`)
  * crt-static-pauthtest.rs

All tests from `assembly-llvm`, `codegen-llvm`, `codegen-units`, `coverage`,
`crashes`, `incremental`, `library`, `mir-opt`, `run-make`, `ui` and
`ui-fulldeps` subsets are expected to pass.

Command to run all passing tests (with tests added by this target explicitly
named for convenience):

```sh
x.py test --target aarch64-unknown-linux-pauthtest --force-rerun assembly-llvm \
  codegen-llvm codegen-units coverage crashes incremental library mir-opt \
  run-make ui ui-fulldeps \
  tests/assembly-llvm/targets/targets-aarch64_unknown_linux_pauthtest.rs \
  tests/codegen-llvm/pauth-attr-special-funcs.rs \
  tests/codegen-llvm/pauth-extern-c.rs \
  tests/codegen-llvm/pauth-extern-c-direct-indirect-call.rs \
  tests/codegen-llvm/pauth-extern-weak-global.rs \
  tests/codegen-llvm/pauth-init-fini.rs \
  tests/run-make/pauth-quicksort-rust-driver \
  tests/run-make/pauth-quicksort-c-driver \
  tests/ui/statics/crt-static-pauthtest.rs
```

## Limitations

Operand bundles should only be attached to indirect function calls. However,
function pointer signing is currently performed in `get_fn_addr`, which causes
the logic to be applied too broadly, including to function values (not just
pointers). As a result, direct calls using signed function values must also
receive operand bundles. Once this is resolved, we should analyze each call and
skip direct calls.
For more information please see the discussion in the [rust-lang issue
tracker](https://github.com/rust-lang/rust/issues/152532).

The current version only supports C interoperability with pointer authentication
features explicitly mentioned at the beginning of this document.

C++ interoperability is not currently supported. Features such as signing C++
member function pointers, virtual function pointers, and virtual table pointers
are not expected to work.
