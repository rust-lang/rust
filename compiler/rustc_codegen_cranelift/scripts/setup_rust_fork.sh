#!/bin/bash
set -e

./y.rs build --no-unstable-features
source scripts/config.sh

echo "[SETUP] Rust fork"
git clone https://github.com/rust-lang/rust.git || true
pushd rust
git fetch
git checkout -- .
git checkout "$(rustc -V | cut -d' ' -f3 | tr -d '(')"

git apply - <<EOF
diff --git a/Cargo.toml b/Cargo.toml
index 5bd1147cad5..10d68a2ff14 100644
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -111,5 +111,7 @@ rustc-std-workspace-std = { path = 'library/rustc-std-workspace-std' }
 rustc-std-workspace-alloc = { path = 'library/rustc-std-workspace-alloc' }
 rustc-std-workspace-std = { path = 'library/rustc-std-workspace-std' }

+compiler_builtins = { path = "../build_sysroot/compiler-builtins" }
+
 [patch."https://github.com/rust-lang/rust-clippy"]
 clippy_lints = { path = "src/tools/clippy/clippy_lints" }
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
EOF

cat > config.toml <<EOF
[llvm]
ninja = false

[build]
rustc = "$(pwd)/../build/bin/cg_clif"
cargo = "$(rustup which cargo)"
full-bootstrap = true
local-rebuild = true

[rust]
codegen-backends = ["cranelift"]
deny-warnings = false
verbose-tests = false
EOF
popd
