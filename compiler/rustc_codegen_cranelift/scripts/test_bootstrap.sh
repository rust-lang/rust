#!/bin/bash
set -e

cd $(dirname "$0")/../

./build.sh
source build/config.sh

echo "[TEST] Bootstrap of rustc"
git clone https://github.com/rust-lang/rust.git || true
pushd rust
git fetch
git checkout -- .
git checkout $(rustc -V | cut -d' ' -f3 | tr -d '(')

git apply - <<EOF
diff --git a/.gitmodules b/.gitmodules
index 984113151de..c1e9d960d56 100644
--- a/.gitmodules
+++ b/.gitmodules
@@ -34,10 +34,6 @@
 [submodule "src/doc/edition-guide"]
 	path = src/doc/edition-guide
 	url = https://github.com/rust-lang/edition-guide.git
-[submodule "src/llvm-project"]
-	path = src/llvm-project
-	url = https://github.com/rust-lang/llvm-project.git
-	branch = rustc/11.0-2020-10-12
 [submodule "src/doc/embedded-book"]
 	path = src/doc/embedded-book
 	url = https://github.com/rust-embedded/book.git
diff --git a/compiler/rustc_data_structures/Cargo.toml b/compiler/rustc_data_structures/Cargo.toml
index 23e689fcae7..5f077b765b6 100644
--- a/compiler/rustc_data_structures/Cargo.toml
+++ b/compiler/rustc_data_structures/Cargo.toml
@@ -32,7 +32,6 @@ tempfile = "3.0.5"

 [dependencies.parking_lot]
 version = "0.11"
-features = ["nightly"]

 [target.'cfg(windows)'.dependencies]
 winapi = { version = "0.3", features = ["fileapi", "psapi"] }
EOF

cat > config.toml <<EOF
[llvm]
ninja = false

[build]
rustc = "$(pwd)/../build/cg_clif"
cargo = "$(rustup which cargo)"
full-bootstrap = true
local-rebuild = true

[rust]
codegen-backends = ["cranelift"]
EOF

rm -r compiler/rustc_codegen_cranelift/{Cargo.*,src}
cp ../Cargo.* compiler/rustc_codegen_cranelift/
cp -r ../src compiler/rustc_codegen_cranelift/src

./x.py build --stage 1 library/std
popd
