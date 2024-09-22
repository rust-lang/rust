#!/usr/bin/env bash
set -e

cd $(dirname "$0")/../

source ./scripts/setup_rust_fork.sh

echo "[TEST] Test suite of rustc"
pushd rust

command -v rg >/dev/null 2>&1 || cargo install ripgrep

rm -r tests/ui/{unsized-locals/,lto/,linkage*} || true
for test in $(rg --files-with-matches "lto" tests/{codegen-units,ui,incremental}); do
  rm $test
done

# should-fail tests don't work when compiletest is compiled with panic=abort
for test in $(rg --files-with-matches "//@ should-fail" tests/{codegen-units,ui,incremental}); do
  rm $test
done

for test in $(rg -i --files-with-matches "//(\[\w+\])?~[^\|]*\s*ERR|//@ error-pattern:|//@(\[.*\])? build-fail|//@(\[.*\])? run-fail|-Cllvm-args" tests/ui); do
  rm $test
done

git checkout -- tests/ui/issues/auxiliary/issue-3136-a.rs # contains //~ERROR, but shouldn't be removed
git checkout -- tests/ui/proc-macro/pretty-print-hack/
git checkout -- tests/ui/entry-point/auxiliary/bad_main_functions.rs
rm tests/ui/parser/unclosed-delimiter-in-dep.rs # submodule contains //~ERROR

# missing features
# ================

# vendor intrinsics
rm tests/ui/asm/x86_64/evex512-implicit-feature.rs # unimplemented AVX512 x86 vendor intrinsic
rm tests/ui/simd/dont-invalid-bitcast-x86_64.rs # unimplemented llvm.x86.sse41.round.ps

# exotic linkages
rm tests/incremental/hashes/function_interfaces.rs
rm tests/incremental/hashes/statics.rs
rm -r tests/run-make/naked-symbol-visibility

# variadic arguments
rm tests/ui/abi/mir/mir_codegen_calls_variadic.rs # requires float varargs
rm tests/ui/abi/variadic-ffi.rs # requires callee side vararg support
rm -r tests/run-make/c-link-to-rust-va-list-fn # requires callee side vararg support
rm tests/ui/delegation/fn-header.rs

# unsized locals
rm -r tests/run-pass-valgrind/unsized-locals

# misc unimplemented things
rm tests/ui/target-feature/missing-plusminus.rs # error not implemented
rm -r tests/run-make/repr128-dwarf # debuginfo test
rm -r tests/run-make/split-debuginfo # same
rm -r tests/run-make/target-specs # i686 not supported by Cranelift
rm -r tests/run-make/mismatching-target-triples # same
rm tests/ui/asm/x86_64/issue-96797.rs # const and sym inline asm operands don't work entirely correctly
rm tests/ui/asm/x86_64/goto.rs # inline asm labels not supported
rm tests/ui/simd/simd-bitmask-notpow2.rs # non-pow-of-2 simd vector sizes
rm -r tests/run-make/embed-source-dwarf # embedding sources in debuginfo

# requires LTO
rm -r tests/run-make/cdylib
rm -r tests/run-make/codegen-options-parsing
rm -r tests/run-make/lto-*
rm -r tests/run-make/reproducible-build-2
rm -r tests/run-make/no-builtins-lto
rm -r tests/run-make/reachable-extern-fn-available-lto

# coverage instrumentation
rm tests/ui/consts/precise-drop-with-coverage.rs
rm tests/ui/issues/issue-85461.rs
rm -r tests/ui/instrument-coverage/

# missing f16/f128 support
rm tests/ui/half-open-range-patterns/half-open-range-pats-semantics.rs
rm tests/ui/asm/aarch64/type-f16.rs
rm tests/ui/float/conv-bits-runtime-const.rs

# optimization tests
# ==================
rm tests/ui/codegen/issue-28950.rs # depends on stack size optimizations
rm tests/ui/codegen/init-large-type.rs # same
rm tests/ui/issues/issue-40883.rs # same
rm -r tests/run-make/fmt-write-bloat/ # tests an optimization
rm tests/ui/statics/const_generics.rs # same

# backend specific tests
# ======================
rm tests/incremental/thinlto/cgu_invalidated_when_import_{added,removed}.rs # requires LLVM
rm -r tests/run-make/cross-lang-lto # same
rm -r tests/run-make/sepcomp-inlining # same
rm -r tests/run-make/sepcomp-separate # same
rm -r tests/run-make/sepcomp-cci-copies # same
rm -r tests/run-make/volatile-intrinsics # same
rm -r tests/run-make/llvm-ident # same
rm -r tests/run-make/no-builtins-attribute # same
rm -r tests/run-make/pgo-gen-no-imp-symbols # same
rm tests/ui/abi/stack-protector.rs # requires stack protector support
rm -r tests/run-make/emit-stack-sizes # requires support for -Z emit-stack-sizes
rm -r tests/run-make/optimization-remarks-dir # remarks are LLVM specific
rm -r tests/run-make/print-to-output # requires --print relocation-models

# requires asm, llvm-ir and/or llvm-bc emit support
# =============================================
rm -r tests/run-make/emit-named-files
rm -r tests/run-make/multiple-emits
rm -r tests/run-make/output-type-permutations
rm -r tests/run-make/emit-to-stdout
rm -r tests/run-make/compressed-debuginfo
rm -r tests/run-make/symbols-include-type-name
rm -r tests/run-make/notify-all-emit-artifacts
rm -r tests/run-make/reset-codegen-1
rm -r tests/run-make/inline-always-many-cgu
rm -r tests/run-make/intrinsic-unreachable

# giving different but possibly correct results
# =============================================
rm tests/ui/mir/mir_misc_casts.rs # depends on deduplication of constants
rm tests/ui/mir/mir_raw_fat_ptr.rs # same
rm tests/ui/consts/issue-33537.rs # same
rm tests/ui/consts/const-mut-refs-crate.rs # same
rm tests/ui/abi/large-byval-align.rs # exceeds implementation limit of Cranelift

# doesn't work due to the way the rustc test suite is invoked.
# should work when using ./x.py test the way it is intended
# ============================================================
rm -r tests/run-make/remap-path-prefix-dwarf # requires llvm-dwarfdump
rm -r tests/run-make/compiler-builtins # Expects lib/rustlib/src/rust to contains the standard library source

# genuine bugs
# ============
rm -r tests/run-make/extern-fn-explicit-align # argument alignment not yet supported
rm -r tests/run-make/panic-abort-eh_frame # .eh_frame emitted with panic=abort
rm tests/ui/deprecation/deprecated_inline_threshold.rs # missing deprecation warning for -Cinline-threshold

# bugs in the test suite
# ======================
rm tests/ui/process/nofile-limit.rs # TODO some AArch64 linking issue
rm tests/ui/backtrace/synchronized-panic-handler.rs # missing needs-unwind annotation
rm tests/ui/lint/non-snake-case/lint-non-snake-case-crate.rs # same
rm -r tests/ui/codegen/equal-pointers-unequal # make incorrect assumptions about the location of stack variables

rm tests/ui/stdio-is-blocking.rs # really slow with unoptimized libstd
rm tests/ui/intrinsics/panic-uninitialized-zeroed.rs # same
rm tests/ui/process/process-panic-after-fork.rs # same

cp ../dist/bin/rustdoc-clif ../dist/bin/rustdoc # some tests expect bin/rustdoc to exist

# prevent $(RUSTDOC) from picking up the sysroot built by x.py. It conflicts with the one used by
# rustdoc-clif
# FIXME remove the bootstrap changes once it is no longer necessary to revert rust-lang/rust#130642
# to avoid building rustc when testing stage0 run-make.
cat <<EOF | git apply -
diff --git a/tests/run-make/tools.mk b/tests/run-make/tools.mk
index ea06b620c4c..b969d0009c6 100644
--- a/tests/run-make/tools.mk
+++ b/tests/run-make/tools.mk
@@ -9,7 +9,7 @@ RUSTC_ORIGINAL := \$(RUSTC)
 BARE_RUSTC := \$(HOST_RPATH_ENV) '\$(RUSTC)'
 BARE_RUSTDOC := \$(HOST_RPATH_ENV) '\$(RUSTDOC)'
 RUSTC := \$(BARE_RUSTC) --out-dir \$(TMPDIR) -L \$(TMPDIR) \$(RUSTFLAGS) -Ainternal_features
-RUSTDOC := \$(BARE_RUSTDOC) -L \$(TARGET_RPATH_DIR)
+RUSTDOC := \$(BARE_RUSTDOC)
 ifdef RUSTC_LINKER
 RUSTC := \$(RUSTC) -Clinker='\$(RUSTC_LINKER)'
 RUSTDOC := \$(RUSTDOC) -Clinker='\$(RUSTC_LINKER)'
diff --git a/src/tools/run-make-support/src/rustdoc.rs b/src/tools/run-make-support/src/rustdoc.rs
index 9607ff02f96..b7d97caf9a2 100644
--- a/src/tools/run-make-support/src/external_deps/rustdoc.rs
+++ b/src/tools/run-make-support/src/external_deps/rustdoc.rs
@@ -34,8 +34,6 @@ pub fn bare() -> Self {
     #[track_caller]
     pub fn new() -> Self {
         let mut cmd = setup_common();
-        let target_rpath_dir = env_var_os("TARGET_RPATH_DIR");
-        cmd.arg(format!("-L{}", target_rpath_dir.to_string_lossy()));
         Self { cmd }
     }

diff --git a/src/bootstrap/src/core/build_steps/test.rs b/src/bootstrap/src/core/build_steps/test.rs
index 2047345d78a..a7e9352bb1c 100644
--- a/src/bootstrap/src/core/build_steps/test.rs
+++ b/src/bootstrap/src/core/build_steps/test.rs
@@ -1733,11 +1733,6 @@ fn run(self, builder: &Builder<'_>) {

         let is_rustdoc = suite.ends_with("rustdoc-ui") || suite.ends_with("rustdoc-js");

-        if mode == "run-make" {
-            let cargo = builder.ensure(tool::Cargo { compiler, target: compiler.host });
-            cmd.arg("--cargo-path").arg(cargo);
-        }
-
         // Avoid depending on rustdoc when we don't need it.
         if mode == "rustdoc"
             || mode == "run-make"
diff --git a/src/tools/compiletest/src/common.rs b/src/tools/compiletest/src/common.rs
index 414f9f3a7f1..5c18179b6fe 100644
--- a/src/tools/compiletest/src/common.rs
+++ b/src/tools/compiletest/src/common.rs
@@ -183,9 +183,6 @@ pub struct Config {
     /// The rustc executable.
     pub rustc_path: PathBuf,

-    /// The cargo executable.
-    pub cargo_path: Option<PathBuf>,
-
     /// The rustdoc executable.
     pub rustdoc_path: Option<PathBuf>,

diff --git a/src/tools/compiletest/src/lib.rs b/src/tools/compiletest/src/lib.rs
index 3339116d542..250b5084d13 100644
--- a/src/tools/compiletest/src/lib.rs
+++ b/src/tools/compiletest/src/lib.rs
@@ -47,7 +47,6 @@ pub fn parse_config(args: Vec<String>) -> Config {
     opts.reqopt("", "compile-lib-path", "path to host shared libraries", "PATH")
         .reqopt("", "run-lib-path", "path to target shared libraries", "PATH")
         .reqopt("", "rustc-path", "path to rustc to use for compiling", "PATH")
-        .optopt("", "cargo-path", "path to cargo to use for compiling", "PATH")
         .optopt("", "rustdoc-path", "path to rustdoc to use for compiling", "PATH")
         .optopt("", "coverage-dump-path", "path to coverage-dump to use in tests", "PATH")
         .reqopt("", "python", "path to python to use for doc tests", "PATH")
@@ -261,7 +260,6 @@ fn make_absolute(path: PathBuf) -> PathBuf {
         compile_lib_path: make_absolute(opt_path(matches, "compile-lib-path")),
         run_lib_path: make_absolute(opt_path(matches, "run-lib-path")),
         rustc_path: opt_path(matches, "rustc-path"),
-        cargo_path: matches.opt_str("cargo-path").map(PathBuf::from),
         rustdoc_path: matches.opt_str("rustdoc-path").map(PathBuf::from),
         coverage_dump_path: matches.opt_str("coverage-dump-path").map(PathBuf::from),
         python: matches.opt_str("python").unwrap(),
@@ -366,7 +364,6 @@ pub fn log_config(config: &Config) {
     logv(c, format!("compile_lib_path: {:?}", config.compile_lib_path));
     logv(c, format!("run_lib_path: {:?}", config.run_lib_path));
     logv(c, format!("rustc_path: {:?}", config.rustc_path.display()));
-    logv(c, format!("cargo_path: {:?}", config.cargo_path));
     logv(c, format!("rustdoc_path: {:?}", config.rustdoc_path));
     logv(c, format!("src_base: {:?}", config.src_base.display()));
     logv(c, format!("build_base: {:?}", config.build_base.display()));
diff --git a/src/tools/compiletest/src/runtest/run_make.rs b/src/tools/compiletest/src/runtest/run_make.rs
index 75fe6a6baaf..852568ae925 100644
--- a/src/tools/compiletest/src/runtest/run_make.rs
+++ b/src/tools/compiletest/src/runtest/run_make.rs
@@ -61,10 +61,6 @@ fn run_rmake_legacy_test(&self) {
             .env_remove("MFLAGS")
             .env_remove("CARGO_MAKEFLAGS");

-        if let Some(ref cargo) = self.config.cargo_path {
-            cmd.env("CARGO", cwd.join(cargo));
-        }
-
         if let Some(ref rustdoc) = self.config.rustdoc_path {
             cmd.env("RUSTDOC", cwd.join(rustdoc));
         }
@@ -413,10 +409,6 @@ fn run_rmake_v2_test(&self) {
             // through a specific CI runner).
             .env("LLVM_COMPONENTS", &self.config.llvm_components);

-        if let Some(ref cargo) = self.config.cargo_path {
-            cmd.env("CARGO", source_root.join(cargo));
-        }
-
         if let Some(ref rustdoc) = self.config.rustdoc_path {
             cmd.env("RUSTDOC", source_root.join(rustdoc));
         }
EOF

echo "[TEST] rustc test suite"
COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 --test-args=--nocapture tests/{codegen-units,run-make,run-pass-valgrind,ui,incremental}
popd
