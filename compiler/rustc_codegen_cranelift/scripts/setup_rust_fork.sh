#!/usr/bin/env bash
set -e

./y.rs build --no-unstable-features

echo "[SETUP] Rust fork"
git clone https://github.com/rust-lang/rust.git || true
pushd rust
git fetch
git checkout -- .
git checkout "$(rustc -V | cut -d' ' -f3 | tr -d '(')"

git apply - <<EOF
diff --git a/library/alloc/Cargo.toml b/library/alloc/Cargo.toml
index d95b5b7f17f..00b6f0e3635 100644
--- a/library/alloc/Cargo.toml
+++ b/library/alloc/Cargo.toml
@@ -8,7 +8,7 @@ edition = "2018"

 [dependencies]
 core = { path = "../core" }
-compiler_builtins = { version = "0.1.40", features = ['rustc-dep-of-std'] }
+compiler_builtins = { version = "0.1.66", features = ['rustc-dep-of-std', 'no-asm'] }

 [dev-dependencies]
 rand = "0.7"
 rand_xorshift = "0.2"
diff --git a/src/tools/compiletest/src/runtest.rs b/src/tools/compiletest/src/runtest.rs
index 8431aa7b818..a3ff7e68ce5 100644
--- a/src/tools/compiletest/src/runtest.rs
+++ b/src/tools/compiletest/src/runtest.rs
@@ -3489,11 +3489,7 @@ fn normalize_output(&self, output: &str, custom_rules: &[(String, String)]) -> S
             .join("library");
         normalize_path(&src_dir, "$(echo '$SRC_DIR')");

-        if let Some(virtual_rust_source_base_dir) =
-            option_env!("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR").map(PathBuf::from)
-        {
-            normalize_path(&virtual_rust_source_base_dir.join("library"), "$(echo '$SRC_DIR')");
-        }
+        normalize_path(&Path::new("$(cd ../build_sysroot/sysroot_src/library; pwd)"), "$(echo '$SRC_DIR')");

         // Paths into the build directory
         let test_build_dir = &self.config.build_base;
EOF

cat > config.toml <<EOF
changelog-seen = 2

[llvm]
ninja = false

[build]
rustc = "$(pwd)/../build/rustc-clif"
cargo = "$(rustup which cargo)"
full-bootstrap = true
local-rebuild = true

[rust]
codegen-backends = ["cranelift"]
deny-warnings = false
verbose-tests = false
EOF
popd
