#!/usr/bin/env bash
set -e

# CG_CLIF_FORCE_GNU_AS will force usage of as instead of the LLVM backend of rustc as
# the LLVM backend isn't compiled in here.
export CG_CLIF_FORCE_GNU_AS=1

# Compiletest expects all standard library paths to start with /rustc/FAKE_PREFIX.
# CG_CLIF_STDLIB_REMAP_PATH_PREFIX will cause cg_clif's build system to pass
# --remap-path-prefix to handle this.
CG_CLIF_STDLIB_REMAP_PATH_PREFIX=/rustc/FAKE_PREFIX ./y.sh build

echo "[SETUP] Rust fork"
git clone --quiet https://github.com/rust-lang/rust.git --filter=tree:0 || true
pushd rust
git fetch
git checkout --no-progress -- .
git checkout --no-progress "$(rustc -V | cut -d' ' -f3 | tr -d '(')"

git submodule update --quiet --init src/tools/cargo library/backtrace library/stdarch

git -c user.name=Dummy -c user.email=dummy@example.com -c commit.gpgSign=false \
    am ../patches/*-stdlib-*.patch

cat > config.toml <<EOF
change-id = 999999

[llvm]
download-ci-llvm = true

[build]
rustc = "$(pwd)/../dist/bin/rustc-clif"
cargo = "$(rustup which cargo)"
full-bootstrap = true
local-rebuild = true
compiletest-allow-stage0 = true

[rust]
download-rustc = false
codegen-backends = ["cranelift"]
deny-warnings = false
verbose-tests = false
# The cg_clif sysroot doesn't contain llvm tools and unless llvm_tools is
# disabled bootstrap will crash trying to copy llvm tools for the bootstrap
# compiler.
llvm-tools = false
std-features = ["panic-unwind"]

EOF

cat <<EOF | git apply -
diff --git a/src/bootstrap/bootstrap.py b/src/bootstrap/bootstrap.py
index 2e16f2cf27..3ac3df99a8 100644
--- a/src/bootstrap/bootstrap.py
+++ b/src/bootstrap/bootstrap.py
@@ -1147,6 +1147,8 @@ class RustBuild(object):
             args += ["-Zwarnings"]
             env["CARGO_BUILD_WARNINGS"] = "deny"

+        env["RUSTFLAGS"] += " -Zbinary-dep-depinfo"
+
         # Add RUSTFLAGS_BOOTSTRAP to RUSTFLAGS for bootstrap compilation.
         # Note that RUSTFLAGS_BOOTSTRAP should always be added to the end of
         # RUSTFLAGS, since that causes RUSTFLAGS_BOOTSTRAP to override RUSTFLAGS.
diff --git a/src/bootstrap/src/core/config/config.rs b/src/bootstrap/src/core/config/config.rs
index bc68bfe396..00143ef3ed 100644
--- a/src/bootstrap/src/core/config/config.rs
+++ b/src/bootstrap/src/core/config/config.rs
@@ -2230,7 +2230,7 @@ pub fn download_ci_rustc_commit<'a>(
                     return None;
                 }

-                if dwn_ctx.is_running_on_ci() {
+                if false && dwn_ctx.is_running_on_ci() {
                     eprintln!("CI rustc commit matches with HEAD and we are in CI.");
                     eprintln!(
                         "\`rustc.download-ci\` functionality will be skipped as artifacts are not available."
diff --git a/src/build_helper/src/git.rs b/src/build_helper/src/git.rs
index 330fb465de..a4593ed96f 100644
--- a/src/build_helper/src/git.rs
+++ b/src/build_helper/src/git.rs
@@ -218,7 +218,7 @@ pub fn get_closest_upstream_commit(
     config: &GitConfig<'_>,
     env: CiEnv,
 ) -> Result<Option<String>, String> {
-    let base = match env {
+    let base = match CiEnv::None {
         CiEnv::None => "HEAD",
         CiEnv::GitHubActions => {
             // On CI, we should always have a non-upstream merge commit at the tip,
EOF

popd

# Allow the testsuite to use llvm tools
host_triple=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
export LLVM_BIN_DIR="$(rustc --print sysroot)/lib/rustlib/$host_triple/bin"
